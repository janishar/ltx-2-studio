// ltx studio — DOM/API helpers shared by every script (loaded first).
// Follows h3 studio's static/util.js.

"use strict";

const $ = (id) => document.getElementById(id);

/** Create an element. attrs: class, text, value, dataset, style (object), on* handlers, properties, attributes. */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "value") node.value = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else if (typeof value !== "string" && key in node) node[key] = value;
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** GET when body is undefined, otherwise POST JSON (the server refuses other content types). */
async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok || (data && data.error)) {
    const error = new Error((data && data.error) || `HTTP ${res.status}`);
    error.data = data;
    throw error;
  }
  return data;
}

function debounce(fn, ms) {
  let timer = null;
  let pending = null;
  const wrapped = (...args) => {
    pending = args;
    clearTimeout(timer);
    timer = setTimeout(() => { const call = pending; pending = null; fn(...call); }, ms);
  };
  wrapped.flush = () => {
    clearTimeout(timer);
    if (pending) { const call = pending; pending = null; return fn(...call); }
    return undefined;
  };
  return wrapped;
}

function fmtSecs(seconds) {
  const s = Math.max(0, Math.round(seconds || 0));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

const randomSeed = () => Math.floor(Math.random() * 2 ** 31);

/** Split an argument string shell-style (single/double quotes, backslash). */
function splitArgs(text) {
  const out = [];
  let current = "", quote = null, has = false;
  for (let i = 0; i < (text || "").length; i++) {
    const c = text[i];
    if (quote) {
      if (c === quote) quote = null;
      else if (c === "\\" && quote === '"' && i + 1 < text.length) current += text[++i];
      else current += c;
    } else if (c === "'" || c === '"') { quote = c; has = true; }
    else if (c === "\\" && i + 1 < text.length) { current += text[++i]; has = true; }
    else if (/\s/.test(c)) { if (has || current) out.push(current); current = ""; has = false; }
    else { current += c; has = true; }
  }
  if (has || current) out.push(current);
  return out;
}

/** Transient notification. kind: "info" | "error" | "ok". Errors stay twice as long. */
function toast(message, { kind = "info", hint = "", timeout = 7000 } = {}) {
  const box = el("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" },
    el("div", { class: "toast-msg", text: message }),
    hint ? el("div", { class: "toast-hint", text: hint }) : null,
    el("button", { class: "toast-close", type: "button", title: "Dismiss", text: "×", onclick: () => box.remove() }));
  $("toasts").append(box);
  while ($("toasts").children.length > 5) $("toasts").firstChild.remove();
  if (timeout) setTimeout(() => box.remove(), kind === "error" ? timeout * 2 : timeout);
  return box;
}

/** True when a keyboard event comes from something the user is typing in. */
function isTyping(target) {
  if (!target) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

/** Close every open ⋮ / ⋯ / history menu except `except`. */
function closePopmenus(except = null) {
  document.querySelectorAll(".popmenu").forEach((menu) => { if (menu !== except) menu.hidden = true; });
  document.querySelectorAll("[aria-haspopup][aria-expanded='true']").forEach((btn) => {
    if (!except || !btn.parentElement.contains(except)) btn.setAttribute("aria-expanded", "false");
  });
}

/** A ⋮-style menu: returns the wrapper holding the toggle button and its popmenu of `items`. */
function popmenu(label, title, items, { align = "right", cls = "ghost" } = {}) {
  const menu = el("div", { class: `popmenu${align === "right" ? " right" : ""}`, role: "menu", hidden: true }, items);
  const button = el("button", {
    class: cls, type: "button", text: label, title, "aria-haspopup": "menu", "aria-expanded": "false",
    onclick: (event) => {
      event.stopPropagation();
      const open = menu.hidden;
      closePopmenus(menu);
      menu.hidden = !open;
      button.setAttribute("aria-expanded", String(open));
    },
  });
  return el("div", { class: "menuwrap" }, button, menu);
}

/** A popmenu item button; errors from `fn` become toasts. */
function menuItem(label, title, fn, cls = "") {
  return el("button", {
    class: cls, type: "button", role: "menuitem", text: label, title,
    onclick: async (event) => {
      event.stopPropagation();
      closePopmenus();
      try { await fn(); } catch (err) { toast(err.message, { kind: "error" }); }
    },
  });
}
