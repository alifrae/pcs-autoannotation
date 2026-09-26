import { test, expect } from "@playwright/test";

const tracePath = process.env.PCS_DAT_E2E_PATH;

test.describe("PCS DAT companion", () => {
  test.skip(!tracePath, "PCS_DAT_E2E_PATH is required for the real DAT smoke");

  test("opens a real DAT and renders camera and LiDAR proposals", async ({ page }) => {
    test.setTimeout(600_000);

    await page.goto("/");
    await page.getByRole("button", { name: "PCS DAT" }).click();

    await expect(page.getByText("PCS DAT Companion")).toBeVisible();
    await expect(page.getByText(/calibration WIP/i)).toBeVisible();

    const pathInput = page.getByPlaceholder("/path/to/trace.dat on the server");
    await pathInput.fill(tracePath!);
    await page.getByRole("button", { name: "Open DAT" }).click();

    await expect(page.getByText("LiDAR points")).toBeVisible({ timeout: 120_000 });
    await expect(page.getByLabel("Camera")).toBeVisible();
    await page.getByLabel("Camera").selectOption("Genicam2");
    await page.getByRole("button", { name: "Load frame" }).click();

    await expect(page.locator("canvas")).toBeVisible({ timeout: 120_000 });
    await expect(page.getByText(/Genicam2/)).toBeVisible();

    await page.getByRole("button", { name: "Run annotations" }).click();

    await expect(
      page.getByText(/proposals · locate-anything-sam2/),
    ).toBeVisible({ timeout: 480_000 });
    await expect(page.getByText("Innov3 LiDAR proposals")).toBeVisible();
    await expect(page.getByText(/3D boxes stay authoritative in PCS/)).toBeVisible();

    const innov3Section = page.getByText("Innov3 LiDAR proposals").locator("..").locator("..");
    await expect(innov3Section.getByText(/proposals/)).toBeVisible();

    await page.screenshot({
      path: "test-results/pcs-dat-workspace.png",
      fullPage: true,
    });
  });
});
