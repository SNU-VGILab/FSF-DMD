(function () {
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------------ tabs / wipe
  function retrigger(el) {
    el.classList.remove("in");
    void el.offsetWidth;
    el.classList.add("in");
  }

  function initWipe() {
    const els = document.querySelectorAll(".wipe");
    if (REDUCED) { els.forEach((e) => e.classList.add("in")); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { threshold: 0.2 });
    els.forEach((e) => io.observe(e));
  }

  function initTabs() {
    document.querySelectorAll("[data-tabs]").forEach((box) => {
      // only this box's own tabs: nested tab groups (appendix) are handled separately
      const own = (e) => e.closest("[data-tabs]") === box;
      const buttons = [...box.querySelectorAll(".tab-list button")].filter(own);
      const panes = [...box.querySelectorAll(".tab-pane")].filter(own);
      const list = buttons.length ? buttons[0].closest(".tab-list") : null;
      const show = (i, scroll) => {
        buttons.forEach((x, j) => x.setAttribute("aria-selected", i === j));
        panes.forEach((p, j) => (p.hidden = i !== j));
        panes[i].querySelectorAll(".wipe").forEach(REDUCED ? (w) => w.classList.add("in") : retrigger);
        if (scroll && list && box.classList.contains("sec-tabs")) {
          const top = box.getBoundingClientRect().top + window.scrollY - 56;
          if (window.scrollY > top) window.scrollTo({ top, behavior: REDUCED ? "auto" : "smooth" });
        }
      };
      buttons.forEach((b, i) => b.addEventListener("click", () => show(i, true)));
    });
  }

  // ------------------------------------------------------------------ Fig. 2 (staged drawing)
  // The panel SVGs are inlined so their elements (tagged .st .s1-.s7) can be revealed stage by stage.
  async function initFig2() {
    const fig = document.getElementById("fig2");
    if (!fig) return;
    const imgs = [...fig.querySelectorAll(".fig2-row img")];
    try {
      const texts = await Promise.all(imgs.map((im) => fetch(im.getAttribute("src")).then((r) => {
        if (!r.ok) throw new Error(r.status);
        return r.text();
      })));
      imgs.forEach((im, i) => {
        const box = document.createElement("div");
        box.innerHTML = texts[i];
        const svg = box.querySelector("svg");
        svg.setAttribute("role", "img");
        svg.setAttribute("aria-label", im.alt);
        im.replaceWith(svg);
      });
    } catch (e) {
      return; // keep the static images
    }
    if (REDUCED) return;
    const play = () => { fig.classList.remove("play"); void fig.offsetWidth; fig.classList.add("play"); };
    fig.classList.add("anim");
    fig.querySelector(".fig2-replay").addEventListener("click", play);
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { io.disconnect(); play(); }
    }, { threshold: 0.3 });
    io.observe(fig);
  }

  // ------------------------------------------------------------------ small pieces
  function initReveal() {
    const els = document.querySelectorAll(".reveal");
    if (REDUCED) { els.forEach((e) => e.classList.add("in")); return; }
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
    els.forEach((e) => io.observe(e));
  }

  function initToc() {
    const links = [...document.querySelectorAll(".toc a")];
    const targets = links.map((a) => document.querySelector(a.getAttribute("href")));
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((a, i) => a.classList.toggle("active", targets[i] === e.target));
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    targets.forEach((t) => t && io.observe(t));
  }

  function initCopy() {
    document.querySelectorAll(".copy-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const text = document.querySelector(btn.dataset.copy).textContent;
        try { await navigator.clipboard.writeText(text); btn.querySelector("span").textContent = "Copied"; }
        catch (e) { btn.querySelector("span").textContent = "Select & copy"; }
        setTimeout(() => (btn.querySelector("span").textContent = "Copy"), 1600);
      });
    });
  }

  function renderMath(root = document.body) {
    if (window.renderMathInElement) {
      renderMathInElement(root, {
        delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }],
        throwOnError: false,
      });
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    renderMath();
    initReveal();
    initWipe();
    initTabs();
    initToc();
    initCopy();
    initFig2();
  });
})();
