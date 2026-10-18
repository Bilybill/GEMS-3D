"use strict";

const menuButton = document.querySelector(".menu-toggle");
const navigation = document.querySelector("#primary-nav");

function closeMenu() {
  menuButton.setAttribute("aria-expanded", "false");
  navigation.classList.remove("is-open");
}

menuButton.addEventListener("click", () => {
  const open = menuButton.getAttribute("aria-expanded") !== "true";
  menuButton.setAttribute("aria-expanded", String(open));
  navigation.classList.toggle("is-open", open);
});
navigation.addEventListener("click", (event) => {
  if (event.target.closest("a")) closeMenu();
});
document.addEventListener("keydown", (event) => {
  if (
    event.key === "Escape" &&
    menuButton.getAttribute("aria-expanded") === "true"
  ) {
    closeMenu();
    menuButton.focus();
  }
});
document.addEventListener("click", (event) => {
  if (!event.target.closest(".nav-wrap")) closeMenu();
});
window.matchMedia("(min-width: 761px)").addEventListener("change", closeMenu);

const modalityTabs = [
  ...document.querySelectorAll('[role="tab"][data-modality]'),
];
const explorer = document.querySelector(".explorer");
const modalityList = document.querySelector(".modality-tabs");
const smallScreen = window.matchMedia("(max-width: 760px)");
function updateTabOrientation() {
  modalityList.setAttribute(
    "aria-orientation",
    smallScreen.matches ? "horizontal" : "vertical",
  );
}
smallScreen.addEventListener("change", updateTabOrientation);
updateTabOrientation();
function selectModality(tab, focus = false) {
  for (const item of modalityTabs) {
    const active = item === tab;
    item.setAttribute("aria-selected", String(active));
    item.tabIndex = active ? 0 : -1;
    document.getElementById(item.getAttribute("aria-controls")).hidden =
      !active;
  }
  explorer.dataset.activeModality = tab.dataset.modality;
  if (focus) tab.focus();
}
for (const tab of modalityTabs) {
  tab.addEventListener("click", () => selectModality(tab));
  tab.addEventListener("keydown", (event) => {
    let index = modalityTabs.indexOf(tab);
    if (event.key === "ArrowRight" || event.key === "ArrowDown") index++;
    else if (event.key === "ArrowLeft" || event.key === "ArrowUp") index--;
    else if (event.key === "Home") index = 0;
    else if (event.key === "End") index = modalityTabs.length - 1;
    else return;
    event.preventDefault();
    selectModality(
      modalityTabs[(index + modalityTabs.length) % modalityTabs.length],
      true,
    );
  });
}

const toast = document.querySelector(".toast");
let toastTimer;
function announce(message) {
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.classList.add("is-visible");
  toastTimer = setTimeout(() => toast.classList.remove("is-visible"), 3000);
}
async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const previousFocus = document.activeElement;
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0";
  document.body.append(area);
  area.select();
  const copied = document.execCommand("copy");
  area.remove();
  previousFocus?.focus({ preventScroll: true });
  if (!copied) throw new Error("Clipboard unavailable");
}
for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", async () => {
    const source = document.getElementById(button.dataset.copy);
    const originalLabel = button.innerHTML;
    button.disabled = true;
    try {
      await copyText(source.textContent.trim());
      button.textContent = "Copied ✓";
      announce("Copied to clipboard.");
      setTimeout(() => {
        button.innerHTML = originalLabel;
      }, 2000);
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(source);
      selection.removeAllRanges();
      selection.addRange(range);
      announce("Select and copy the highlighted text.");
    } finally {
      button.disabled = false;
    }
  });
}

// Keep the current section visible in the desktop navigation.
const sectionLinks = [...navigation.querySelectorAll('a[href^="#"]')];
const readingProgress = document.querySelector(".reading-progress");
const topLink = document.querySelector(".back-to-top");
let readingFrame;
function updateReadingPosition() {
  readingFrame = undefined;
  const maxScroll = document.documentElement.scrollHeight - innerHeight;
  const progress =
    maxScroll > 0 ? Math.min(1, Math.max(0, scrollY / maxScroll)) : 0;
  readingProgress.style.transform = `scaleX(${progress})`;
  topLink.hidden = scrollY < 700;
  const threshold = document.querySelector(".site-header").offsetHeight + 120;
  let active;
  for (const link of sectionLinks) {
    if (
      document.querySelector(link.hash).getBoundingClientRect().top <= threshold
    )
      active = link;
  }
  for (const link of sectionLinks) {
    if (link === active) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  }
}
function scheduleReadingPosition() {
  if (!readingFrame)
    readingFrame = requestAnimationFrame(updateReadingPosition);
}
window.addEventListener("scroll", scheduleReadingPosition, { passive: true });
window.addEventListener("resize", scheduleReadingPosition, { passive: true });
window.addEventListener("load", scheduleReadingPosition);
updateReadingPosition();

const figureTabs = [...document.querySelectorAll("[data-figure-tab]")];
function selectFigure(tab, focus = false) {
  for (const item of figureTabs) {
    const active = item === tab;
    item.setAttribute("aria-selected", String(active));
    item.tabIndex = active ? 0 : -1;
    document.getElementById(item.getAttribute("aria-controls")).hidden =
      !active;
  }
  if (focus) tab.focus();
  scheduleReadingPosition();
}
for (const tab of figureTabs) {
  tab.addEventListener("click", () => selectFigure(tab));
  tab.addEventListener("keydown", (event) => {
    let index = figureTabs.indexOf(tab);
    if (event.key === "ArrowRight") index++;
    else if (event.key === "ArrowLeft") index--;
    else if (event.key === "Home") index = 0;
    else if (event.key === "End") index = figureTabs.length - 1;
    else return;
    event.preventDefault();
    selectFigure(
      figureTabs[(index + figureTabs.length) % figureTabs.length],
      true,
    );
  });
}

// Inspect source figures without leaving the project page.
const viewer = document.querySelector(".figure-viewer");
const viewerImage = viewer.querySelector(".viewer-image");
const viewerStage = viewer.querySelector(".viewer-stage");
const viewerClose = viewer.querySelector(".viewer-close");
const zoomLevel = viewer.querySelector(".zoom-level");
const viewerLoading = viewer.querySelector(".viewer-loading");
let viewerOpener;
let imageScale = 1;
let fitScale = 1;
function setImageScale(value, preserveCenter = true) {
  if (!viewerImage.naturalWidth) return;
  const previous = imageScale;
  imageScale = Math.min(3, Math.max(Math.min(fitScale, 0.1), value));
  viewerImage.style.width = `${Math.round(viewerImage.naturalWidth * imageScale)}px`;
  zoomLevel.textContent = `${Math.round(imageScale * 100)}%`;
  viewer.querySelector('[data-zoom="in"]').disabled = imageScale >= 3;
  viewer.querySelector('[data-zoom="out"]').disabled =
    imageScale <= Math.min(fitScale, 0.1);
  if (preserveCenter) {
    const ratio = imageScale / previous;
    viewerStage.scrollLeft =
      (viewerStage.scrollLeft + viewerStage.clientWidth / 2) * ratio -
      viewerStage.clientWidth / 2;
    viewerStage.scrollTop =
      (viewerStage.scrollTop + viewerStage.clientHeight / 2) * ratio -
      viewerStage.clientHeight / 2;
  }
}
function fitImage() {
  if (!viewer.open || !viewerImage.naturalWidth) return;
  fitScale = Math.min(
    1,
    (viewerStage.clientWidth - 48) / viewerImage.naturalWidth,
    (viewerStage.clientHeight - 48) / viewerImage.naturalHeight,
  );
  setImageScale(fitScale, false);
  viewerStage.scrollTo(0, 0);
}
for (const link of document.querySelectorAll("[data-lightbox]")) {
  link.addEventListener("click", (event) => {
    if (
      !viewer.showModal ||
      event.ctrlKey ||
      event.metaKey ||
      event.shiftKey ||
      event.altKey
    )
      return;
    event.preventDefault();
    viewerOpener = link;
    document.querySelector("#viewer-title").textContent = link.dataset.title;
    document.querySelector("#viewer-description").textContent =
      link.dataset.caption;
    viewerImage.alt = link.querySelector("img").alt;
    viewerLoading.textContent = "Loading high-resolution figure…";
    viewerLoading.hidden = false;
    viewerImage.style.visibility = "hidden";
    viewerImage.onload = async () => {
      try {
        await viewerImage.decode();
      } catch {
        /* onerror handles failed loads */
      }
      viewerImage.style.visibility = "visible";
      viewerLoading.hidden = true;
      fitImage();
    };
    viewerImage.onerror = () => {
      viewerLoading.textContent =
        "This image could not load. Close this view and open the original PDF.";
      viewerLoading.hidden = false;
    };
    viewerImage.src = link.href;
    viewer.showModal();
    document.body.classList.add("has-image-viewer");
    viewerClose.focus({ preventScroll: true });
    if (viewerImage.complete && viewerImage.naturalWidth) {
      viewerImage.style.visibility = "visible";
      viewerLoading.hidden = true;
      fitImage();
    }
  });
}
viewerClose.addEventListener("click", () => viewer.close());
viewer.addEventListener("close", () => {
  document.body.classList.remove("has-image-viewer");
  viewerOpener?.focus({ preventScroll: true });
});
viewer.addEventListener("click", (event) => {
  if (event.target !== viewer) return;
  const bounds = viewer.getBoundingClientRect();
  if (
    event.clientX < bounds.left ||
    event.clientX > bounds.right ||
    event.clientY < bounds.top ||
    event.clientY > bounds.bottom
  )
    viewer.close();
});
viewer.addEventListener("keydown", (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key === "+" || event.key === "=") {
    event.preventDefault();
    setImageScale(imageScale * 1.25);
  } else if (event.key === "-") {
    event.preventDefault();
    setImageScale(imageScale / 1.25);
  } else if (event.key === "0") {
    event.preventDefault();
    fitImage();
  }
});
for (const button of viewer.querySelectorAll("[data-zoom]")) {
  button.addEventListener("click", () => {
    if (button.dataset.zoom === "in") setImageScale(imageScale * 1.25);
    else if (button.dataset.zoom === "out") setImageScale(imageScale / 1.25);
    else if (button.dataset.zoom === "actual") setImageScale(1);
    else fitImage();
  });
}
window.addEventListener("resize", () => {
  if (viewer.open) fitImage();
});
