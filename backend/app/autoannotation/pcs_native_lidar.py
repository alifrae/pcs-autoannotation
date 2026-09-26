from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

_IFSCAN10_PRE_MAGIC = 0x00AC
_IFSCAN10_INTERFACE_TYPE = 0x0010
_IFSCAN10_ANGLE_SCALE_RAD = np.float32(9.58738e-05)
_IFSCAN10_DISTANCE_SIGNAL_MASK = np.uint16(0x7FFF)
_IFSCAN10_DISTANCE_INVALID = np.uint16(0xFFFF)
_IFSCAN10_ENERGY_SIGNAL_MASK = np.uint16(0x1FFF)
_IFSCAN10_ENERGY_RESERVED = np.uint16(0x1FFF)


def is_ifscan10_payload(native: Any, payload: bytes) -> bool:
    try:
        pre_magic, interface_type = native.export.extract_ifscan_magic_word(payload)
        major, _minor = native.export.extract_ifscan_interface_version(payload)
    except (AttributeError, TypeError, ValueError):
        return False
    return (
        int(pre_magic) == _IFSCAN10_PRE_MAGIC
        and int(interface_type) == _IFSCAN10_INTERFACE_TYPE
        and int(major) == 10
    )


def extract_ifscan10_session(native: Any, payload: bytes) -> Any:
    extract = native.export.extract_ifscan10_session_config
    try:
        return extract(payload, protocol="ifscan10")
    except TypeError:
        return extract(payload)


def decode_point_cloud_payload(
    native: Any,
    payload: bytes,
    *,
    ifscan10_session: Any | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Decode a PCS-native point-cloud payload without duplicating binary parsing."""

    if is_ifscan10_payload(native, payload):
        session = ifscan10_session or extract_ifscan10_session(native, payload)
        return _decode_ifscan10(native, payload, session=session)

    decoded = native.codec.decode_ifscan_payload(payload)
    points = np.ascontiguousarray(np.asarray(decoded.points, dtype=np.float32))
    attributes: dict[str, np.ndarray] = {}
    for native_name, canonical_name in (
        ("range", "range"),
        ("intensity", "intensity"),
        ("reflectivity", "reflectivity"),
        ("inferred_reflectivity", "inferred_reflectivity"),
        ("slot_index", "slot_index"),
        ("layer_index", "layer_index"),
        ("echo_index", "echo_index"),
        ("peak", "peak"),
        ("energy", "energy"),
        ("width", "width"),
        ("peak_width", "peak_width"),
        ("flags", "flags"),
    ):
        value = getattr(decoded, native_name, None)
        if value is not None:
            attributes[canonical_name] = np.ascontiguousarray(np.asarray(value))
    metadata = {
        "decoder_path": "pcs_native_legacy_structured",
        "ifscan_version": getattr(decoded, "ifscan_version", None),
        "structure_kind": getattr(decoded, "structure_kind", None),
        "num_slots": getattr(decoded, "num_slots", None),
        "num_layers": getattr(decoded, "num_layers", None),
        "num_echoes": getattr(decoded, "num_echoes", None),
        "valid_count": getattr(decoded, "valid_count", None),
        "housing_merged": False,
    }
    return points, attributes, metadata


def _decode_ifscan10(
    native: Any,
    payload: bytes,
    *,
    session: Any,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    capsules = native.export.decode_ifscan10_to_arrow(payload, session)
    slots = _import_arrow_batch(capsules["slots"])
    internal_housing = (
        _import_arrow_batch(capsules["internal_housing"])
        if capsules.get("internal_housing") is not None
        else None
    )

    num_slots = int(session.num_slots)
    num_layers = int(session.num_layers)
    num_echoes = int(session.num_echos)
    if min(num_slots, num_layers, num_echoes) <= 0:
        raise RuntimeError(
            "PCS IFSCAN10 session exposes invalid topology: "
            f"slots={num_slots}, layers={num_layers}, echoes={num_echoes}"
        )

    distance_raw = _reshape_sle(
        _batch_column(slots, "distance", "radDistCm", "radDistCm_LRflag"),
        num_slots=num_slots,
        num_layers=num_layers,
        num_echoes=num_echoes,
        dtype=np.uint16,
    )
    if distance_raw is None:
        raise RuntimeError("PCS IFSCAN10 Arrow output has no complete distance grid")
    distance_cm, invalid = _decode_distance(distance_raw)

    horizontal = _angle_grid(
        _batch_column(slots, "horizontal_angle", "horizontalAngle"),
        num_slots=num_slots,
        num_layers=num_layers,
        name="horizontal angle",
    )
    vertical = _angle_grid(
        _batch_column(slots, "vertical_angle", "verticalAngle"),
        num_slots=num_slots,
        num_layers=num_layers,
        name="vertical angle",
    )
    energy_sle = _decode_energy(
        _reshape_sle(
            _batch_column(slots, "energy", "intensity"),
            num_slots=num_slots,
            num_layers=num_layers,
            num_echoes=num_echoes,
        )
    )
    peak_sle = _reshape_sle(
        _batch_column(slots, "peak", "peakAmplitude", "peak_amp"),
        num_slots=num_slots,
        num_layers=num_layers,
        num_echoes=num_echoes,
    )
    reflectivity_sle = _reshape_sle(
        _batch_column(slots, "reflectivity"),
        num_slots=num_slots,
        num_layers=num_layers,
        num_echoes=num_echoes,
        dtype=np.uint8,
    )
    housing = _housing_grids(
        internal_housing,
        num_slots=num_slots,
        num_layers=num_layers,
    )

    housing_kwargs: dict[str, Any] = {}
    if housing is not None:
        housing_kwargs = {
            "housing_start_of_echo": housing["startOfEcho"],
            "housing_start_fwhm": housing.get("startOfFWHM_integer"),
            "housing_end_fwhm": housing.get("endOfFWHM_integer"),
            "housing_flags": housing.get("flags"),
            "housing_distance_factor": 15.0,
            "housing_fwhm_factor": 0.5,
        }

    compacted = native.math.compact_ifscan10_spherical(
        np.ascontiguousarray(distance_cm, dtype=np.float32),
        np.ascontiguousarray(invalid, dtype=bool),
        np.ascontiguousarray(horizontal, dtype=np.float32),
        np.ascontiguousarray(vertical, dtype=np.float32),
        None if energy_sle is None else np.ascontiguousarray(energy_sle, dtype=np.float32),
        None
        if reflectivity_sle is None
        else np.ascontiguousarray(reflectivity_sle, dtype=np.uint8),
        distance_min_cm=1.0,
        distance_max_cm=30000.0,
        **housing_kwargs,
    )

    points = np.ascontiguousarray(np.asarray(compacted["points"], dtype=np.float32))
    slot_index = np.ascontiguousarray(np.asarray(compacted["slot_index"], dtype=np.int32))
    layer_index = np.ascontiguousarray(np.asarray(compacted["layer_index"], dtype=np.int32))
    echo_index = np.ascontiguousarray(np.asarray(compacted["echo_index"], dtype=np.int32))
    housing_merged = bool(compacted.get("housing_merged", False))

    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"PCS IFSCAN10 points have unexpected shape {points.shape}")
    if not (len(points) == len(slot_index) == len(layer_index) == len(echo_index)):
        raise RuntimeError("PCS IFSCAN10 compacted point/index arrays have inconsistent lengths")

    attributes: dict[str, np.ndarray] = {
        "range": np.ascontiguousarray(np.asarray(compacted["range"], dtype=np.float32)),
        "slot_index": slot_index,
        "layer_index": layer_index,
        "echo_index": echo_index,
    }

    peak = _gather_point_attribute(
        peak_sle,
        slot_index=slot_index,
        layer_index=layer_index,
        echo_index=echo_index,
        housing_merged=housing_merged,
        housing_values=None if housing is None else housing.get("peakAmplitude"),
    )
    if peak is not None:
        attributes["peak"] = peak
        attributes["intensity"] = peak

    energy = _gather_point_attribute(
        energy_sle,
        slot_index=slot_index,
        layer_index=layer_index,
        echo_index=echo_index,
        housing_merged=housing_merged,
    )
    if energy is not None:
        attributes["energy"] = energy

    reflectivity = compacted.get("reflectivity")
    if reflectivity is not None:
        attributes["reflectivity"] = np.ascontiguousarray(
            np.asarray(reflectivity, dtype=np.uint8)
        )

    metadata = {
        "decoder_path": "pcs_native_ifscan10_arrow_geometry",
        "ifscan_version": 10,
        "structure_kind": "ifscan10",
        "num_slots": num_slots,
        "num_layers": num_layers,
        "num_echoes": num_echoes,
        "num_mirrors": int(getattr(session, "num_mirrors", 0) or 0),
        "content_type": int(getattr(session, "content_type", 0) or 0),
        "active_data_element": int(getattr(session, "active_data_element", 0) or 0),
        "slot_layout": str(getattr(session, "slot_layout", "") or ""),
        "variant": str(getattr(session, "variant", "") or ""),
        "housing_merged": housing_merged,
        "valid_count": int(len(points)),
    }
    return points, attributes, metadata


def _import_arrow_batch(capsule_pair: tuple[Any, Any]) -> Any:
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError(
            "PCS IFSCAN10 decoding requires PyArrow for the native Arrow C interface"
        ) from exc

    array_capsule, schema_capsule = capsule_pair
    importer = getattr(pa.RecordBatch, "_import_from_c", None)
    if importer is None:
        raise RuntimeError("installed PyArrow does not support RecordBatch._import_from_c")

    if (
        type(array_capsule).__name__ == "PyCapsule"
        and type(schema_capsule).__name__ == "PyCapsule"
    ):
        get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
        get_ptr.restype = ctypes.c_void_p
        get_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
        array_ptr = get_ptr(array_capsule, b"arrow_array")
        schema_ptr = get_ptr(schema_capsule, b"arrow_schema")
        if not array_ptr or not schema_ptr:
            raise RuntimeError("failed to extract Arrow C pointers from PCS native capsules")
        return importer(array_ptr, schema_ptr)

    return importer(array_capsule, schema_capsule)


def _batch_column(batch: Any | None, *names: str) -> np.ndarray | None:
    if batch is None:
        return None
    schema_names = set(getattr(getattr(batch, "schema", None), "names", []) or [])
    for name in names:
        if name in schema_names:
            return np.asarray(batch[name].to_numpy(zero_copy_only=False))
    return None


def _reshape_sle(
    values: np.ndarray | None,
    *,
    num_slots: int,
    num_layers: int,
    num_echoes: int,
    dtype: Any | None = None,
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    if array.size != num_slots * num_layers * num_echoes:
        return None
    return np.ascontiguousarray(
        array.reshape(num_slots, num_echoes, num_layers).transpose(0, 2, 1)
    )


def _reshape_grid(
    values: np.ndarray | None,
    *,
    num_slots: int,
    num_layers: int,
    dtype: Any | None = None,
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    if array.size != num_slots * num_layers:
        return None
    return np.ascontiguousarray(array.reshape(num_slots, num_layers))


def _angle_grid(
    values: np.ndarray | None,
    *,
    num_slots: int,
    num_layers: int,
    name: str,
) -> np.ndarray:
    if values is None:
        raise RuntimeError(f"PCS IFSCAN10 Arrow output is missing {name}")
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.floating):
        values_f = array.astype(np.float32, copy=False)
    else:
        values_f = array.astype(np.int16, copy=False).astype(np.float32)
        values_f *= _IFSCAN10_ANGLE_SCALE_RAD

    if values_f.size == num_slots * num_layers:
        return np.ascontiguousarray(values_f.reshape(num_slots, num_layers))
    if values_f.size == num_layers:
        return np.ascontiguousarray(np.repeat(values_f[None, :], num_slots, axis=0))
    raise RuntimeError(
        f"PCS IFSCAN10 {name} has unexpected size {values_f.size}; "
        f"expected {num_slots * num_layers} or {num_layers}"
    )


def _decode_distance(raw_sle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_sle, dtype=np.uint16)
    signal = np.bitwise_and(raw, _IFSCAN10_DISTANCE_SIGNAL_MASK)
    invalid = (raw == _IFSCAN10_DISTANCE_INVALID) | (
        signal == _IFSCAN10_DISTANCE_SIGNAL_MASK
    )
    return (
        np.ascontiguousarray(signal.astype(np.float32)),
        np.ascontiguousarray(invalid),
    )


def _decode_energy(raw_sle: np.ndarray | None) -> np.ndarray | None:
    if raw_sle is None:
        return None
    raw = np.asarray(raw_sle)
    if np.issubdtype(raw.dtype, np.integer):
        signal = np.bitwise_and(raw.astype(np.uint16, copy=False), _IFSCAN10_ENERGY_SIGNAL_MASK)
        signal = np.where(signal == _IFSCAN10_ENERGY_RESERVED, np.uint16(0), signal)
        return np.ascontiguousarray(signal.astype(np.float32))
    return np.ascontiguousarray(raw.astype(np.float32, copy=False))


def _housing_grids(
    batch: Any | None,
    *,
    num_slots: int,
    num_layers: int,
) -> dict[str, np.ndarray] | None:
    if batch is None:
        return None
    start = _reshape_grid(
        _batch_column(batch, "startOfEcho"),
        num_slots=num_slots,
        num_layers=num_layers,
        dtype=np.uint16,
    )
    peak = _reshape_grid(
        _batch_column(batch, "peakAmplitude", "peak", "peakAmp"),
        num_slots=num_slots,
        num_layers=num_layers,
        dtype=np.uint8,
    )
    if start is None or peak is None:
        return None
    result = {"startOfEcho": start, "peakAmplitude": peak}
    for output_name, aliases, dtype in (
        ("startOfFWHM_integer", ("startOfFWHM_integer",), np.uint8),
        ("endOfFWHM_integer", ("endOfFWHM_integer",), np.uint8),
        ("flags", ("flags",), np.uint8),
    ):
        grid = _reshape_grid(
            _batch_column(batch, *aliases),
            num_slots=num_slots,
            num_layers=num_layers,
            dtype=dtype,
        )
        if grid is not None:
            result[output_name] = grid
    return result


def _gather_point_attribute(
    values_sle: np.ndarray | None,
    *,
    slot_index: np.ndarray,
    layer_index: np.ndarray,
    echo_index: np.ndarray,
    housing_merged: bool,
    housing_values: np.ndarray | None = None,
) -> np.ndarray | None:
    if values_sle is None and housing_values is None:
        return None
    output = np.zeros(len(slot_index), dtype=np.float32)
    slots = np.asarray(slot_index, dtype=np.int64)
    layers = np.asarray(layer_index, dtype=np.int64)
    echoes = np.asarray(echo_index, dtype=np.int64)

    if housing_merged and housing_values is not None:
        mask = echoes == 0
        if np.any(mask):
            output[mask] = np.asarray(housing_values)[slots[mask], layers[mask]]

    if values_sle is not None:
        source_echo = echoes - 1 if housing_merged else echoes
        mask = (source_echo >= 0) & (source_echo < values_sle.shape[2])
        if np.any(mask):
            output[mask] = values_sle[
                slots[mask],
                layers[mask],
                source_echo[mask],
            ]
    return np.ascontiguousarray(output)
