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
