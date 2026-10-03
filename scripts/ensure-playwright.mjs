import { execFile } from "node:child_process";
import { access } from "node:fs/promises";
import { promisify } from "node:util";
import { chromium } from "playwright";

try {
  await access(chromium.executablePath());
} catch {
  await promisify(execFile)("npx", ["playwright", "install", "chromium"], {
    stdio: "inherit",
  });
}
