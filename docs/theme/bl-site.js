/* BatchLAS site chrome: left navigation, "on this page" table of contents and full-text
   search. Data comes from bl-nav.js and bl-search.js, which docs/tools/gen_site_assets.py
   writes after Doxygen runs. Plain script (no modules) so the site also works from file://. */
(function () {
  "use strict";

  var here = (location.pathname.split("/").pop() || "index.html").replace(/[?#].*$/, "");

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  /* ------------------------------------------------------------ navigation */
  function containsHere(node) {
    if (node.u === here) return true;
    return (node.c || []).some(containsHere);
  }

  function navList(nodes, depth) {
    var ul = el("ul", "bl-nav-list");
    nodes.forEach(function (n) {
      var li = el("li", "bl-nav-item");
      var row = el("div", "bl-nav-row");
      var a = el("a", "bl-nav-link", n.t);
      a.href = n.u;
      if (n.u === here) { a.classList.add("bl-current"); a.setAttribute("aria-current", "page"); }
      row.appendChild(a);
      li.appendChild(row);
      if (n.c && n.c.length) {
        var open = containsHere(n) || (depth === 0 && n.u === here);
        var btn = el("button", "bl-nav-toggle-sub");
        btn.type = "button";
        btn.setAttribute("aria-label", "Expand " + n.t);
        btn.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M8.6 16.6 13.2 12 8.6 7.4 10 6l6 6-6 6z"/></svg>';
        row.appendChild(btn);
        var sub = navList(n.c, depth + 1);
        li.appendChild(sub);
        if (open) li.classList.add("bl-open");
        btn.addEventListener("click", function () { li.classList.toggle("bl-open"); });
      }
      ul.appendChild(li);
    });
    return ul;
  }

  function buildNav() {
    if (!window.BL_NAV) return;
    var nav = el("nav", "bl-nav");
    nav.setAttribute("aria-label", "Documentation");
    nav.appendChild(el("div", "bl-nav-title", "BatchLAS documentation"));
    nav.appendChild(navList(window.BL_NAV, 0));
    document.body.appendChild(nav);
    var cur = nav.querySelector(".bl-current");
    if (cur) cur.scrollIntoView({ block: "center" });
    var scrim = el("div", "bl-scrim");
    scrim.addEventListener("click", function () { document.documentElement.classList.remove("bl-nav-open"); });
    document.body.appendChild(scrim);
  }

  /* ------------------------------------------------------- table of contents */
  function buildToc() {
    var contents = document.querySelector("div.contents");
    if (!contents) return;
    var heads = contents.querySelectorAll(
      "h1.doxsection, h2.doxsection, h2.groupheader");
    var items = [];
    heads.forEach(function (h) {
      var a = h.querySelector("a.anchor[id]") || h.querySelector("a[id]");
      var id = a ? a.id : h.id;
      if (!id) return;
      var text = h.textContent.replace(/\s+/g, " ").trim();
      if (!text) return;
      var level = h.classList.contains("groupheader") ? 1 : (h.tagName === "H1" ? 1 : 2);
      items.push({ id: id, text: text, level: level, head: h });
    });
    if (items.length < 2) { document.documentElement.classList.add("bl-no-toc"); return; }
    var toc = el("nav", "bl-toc");
    toc.setAttribute("aria-label", "On this page");
    toc.appendChild(el("div", "bl-toc-title", "Table of contents"));
    var ul = el("ul", "bl-toc-list");
    var links = {};
    items.forEach(function (it) {
      var li = el("li", "bl-toc-l" + it.level);
      var a = el("a", null, it.text);
      a.href = "#" + it.id;
      links[it.id] = a;
      li.appendChild(a);
      ul.appendChild(li);
    });
    toc.appendChild(ul);
    document.body.appendChild(toc);

    var active = null;
    function spy() {
      var y = window.scrollY + 120, cur = items[0];
      for (var i = 0; i < items.length; ++i) {
        if (items[i].head.getBoundingClientRect().top + window.scrollY <= y) cur = items[i];
        else break;
      }
      if (cur && links[cur.id] !== active) {
        if (active) active.classList.remove("bl-active");
        active = links[cur.id];
        active.classList.add("bl-active");
      }
    }
    var pending = false;
    window.addEventListener("scroll", function () {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function () { pending = false; spy(); });
    }, { passive: true });
    spy();
  }

  /* ------------------------------------------------------------------ search */
  var index = null, loading = false, waiters = [];
  function loadIndex(cb) {
    if (index) return cb(index);
    waiters.push(cb);
    if (loading) return;
    loading = true;
    var s = document.createElement("script");
    s.src = "bl-search.js" + (window.BL_SEARCH_V ? "?v=" + window.BL_SEARCH_V : "");
    s.onload = function () {
      index = (window.BL_SEARCH || []).map(function (r) {
        return { r: r, t: r.t.toLowerCase(), p: r.p.toLowerCase(), x: r.x.toLowerCase() };
      });
      waiters.splice(0).forEach(function (w) { w(index); });
    };
    document.head.appendChild(s);
  }

  function esc(s) {
    return s.replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; });
  }

  function highlight(text, terms) {
    var out = esc(text);
    terms.forEach(function (t) {
      if (t.length < 2) return;
      out = out.replace(new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig"), "\u0001$1\u0002");
    });
    return out.replace(/\u0001/g, "<mark>").replace(/\u0002/g, "</mark>");
  }

  function snippet(text, lower, terms) {
    var at = -1;
    terms.forEach(function (t) { var i = lower.indexOf(t); if (i >= 0 && (at < 0 || i < at)) at = i; });
    if (at < 0) at = 0;
    var start = Math.max(0, at - 60);
    if (start > 0) { var sp = text.indexOf(" ", start); if (sp > 0 && sp < at) start = sp + 1; }
    var cut = text.slice(start, start + 220);
    return (start > 0 ? "… " : "") + cut + (start + 220 < text.length ? " …" : "");
  }

  function count(hay, needle) {
    var n = 0, i = hay.indexOf(needle);
    while (i >= 0 && n < 8) { ++n; i = hay.indexOf(needle, i + needle.length); }
    return n;
  }

  function search(q) {
    var terms = q.toLowerCase().split(/[^\w:~]+/).filter(Boolean);
    if (!terms.length) return [];
    var pages = {}, order = [];
    index.forEach(function (e) {
      var score = 0;
      for (var i = 0; i < terms.length; ++i) {
        var t = terms[i];
        var s = (e.t.indexOf(t) >= 0 ? 12 : 0) + (e.p.indexOf(t) >= 0 ? 4 : 0) + count(e.x, t);
        if (!s) return;
        score += s;
      }
      if (e.r.k) score *= 0.6;
      var key = e.r.u.split("#")[0];
      var pg = pages[key];
      if (!pg) { pg = pages[key] = { title: e.r.p, url: key, hits: [], best: 0 }; order.push(pg); }
      pg.hits.push({ e: e, score: score });
      pg.best = Math.max(pg.best, score);
    });
    order.forEach(function (pg) {
      pg.hits.sort(function (a, b) { return b.score - a.score; });
      pg.rank = pg.best + Math.min(pg.hits.length, 10) * 0.5;
    });
    order.sort(function (a, b) { return b.rank - a.rank; });
    return { pages: order, terms: terms };
  }

  var ICON_PAGE = '<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path fill="currentColor" d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm4 18H6V4h7v5h5v11zM8 12h8v2H8zm0 4h8v2H8z"/></svg>';

  function renderResults(box, q) {
    var res = search(q);
    box.innerHTML = "";
    if (!q.trim()) { box.appendChild(el("div", "bl-sr-meta", "Type to start searching")); return; }
    var pages = res.pages || [];
    box.appendChild(el("div", "bl-sr-meta", pages.length ? pages.length + " matching document" + (pages.length === 1 ? "" : "s") : "No matching documents"));
    pages.slice(0, 40).forEach(function (pg) {
      var item = el("div", "bl-sr-item");
      var first = pg.hits[0].e;
      var head = el("a", "bl-sr-page");
      head.href = pg.url;
      head.innerHTML = ICON_PAGE + "<span>" + highlight(pg.title, res.terms) + "</span>";
      item.appendChild(head);
      function sectionLink(h) {
        var a = el("a", "bl-sr-section");
        a.href = h.e.r.u;
        a.innerHTML = (h.e.r.t ? '<span class="bl-sr-stitle">' + highlight(h.e.r.t, res.terms) + "</span>" : "") +
          '<span class="bl-sr-text">' + highlight(snippet(h.e.r.x, h.e.x, res.terms), res.terms) + "</span>";
        return a;
      }
      item.appendChild(sectionLink(pg.hits[0]));
      if (pg.hits.length > 1) {
        var more = el("button", "bl-sr-more", (pg.hits.length - 1) + " more on this page");
        more.type = "button";
        var rest = el("div", "bl-sr-rest");
        pg.hits.slice(1, 12).forEach(function (h) { rest.appendChild(sectionLink(h)); });
        more.addEventListener("click", function () { item.classList.toggle("bl-sr-expanded"); });
        item.appendChild(more);
        item.appendChild(rest);
      }
      box.appendChild(item);
    });
  }

  function buildSearch() {
    var root = document.querySelector(".bl-search");
    if (!root) return;
    var input = root.querySelector("input");
    var box = root.querySelector(".bl-search-results");
    var html = document.documentElement;
    var timer = null;
    function open() {
      html.classList.add("bl-search-active");
      loadIndex(function () { renderResults(box, input.value); });
    }
    function close() { html.classList.remove("bl-search-active"); input.blur(); }
    input.addEventListener("focus", open);
    input.addEventListener("input", function () {
      clearTimeout(timer);
      timer = setTimeout(function () { loadIndex(function () { renderResults(box, input.value); }); }, 80);
    });
    input.addEventListener("keydown", function (ev) {
      var links = Array.prototype.slice.call(box.querySelectorAll("a"));
      if (ev.key === "Escape") { close(); }
      else if (ev.key === "Enter" && links.length) { location.href = links[0].href; close(); }
      else if (ev.key === "ArrowDown" && links.length) { ev.preventDefault(); links[0].focus(); }
    });
    box.addEventListener("keydown", function (ev) {
      var links = Array.prototype.slice.call(box.querySelectorAll("a"));
      var i = links.indexOf(document.activeElement);
      if (ev.key === "ArrowDown" && i < links.length - 1) { ev.preventDefault(); links[i + 1].focus(); }
      else if (ev.key === "ArrowUp") { ev.preventDefault(); (i > 0 ? links[i - 1] : input).focus(); }
      else if (ev.key === "Escape") close();
    });
    box.addEventListener("click", function (ev) { if (ev.target.closest("a")) close(); });
    var reset = root.querySelector(".bl-search-reset");
    if (reset) reset.addEventListener("click", function () { input.value = ""; input.focus(); renderResults(box, ""); });
    var overlay = el("div", "bl-search-overlay");
    overlay.addEventListener("click", close);
    document.body.appendChild(overlay);
    document.addEventListener("keydown", function (ev) {
      var t = ev.target.tagName;
      if ((ev.key === "/" || ev.key === "s") && t !== "INPUT" && t !== "TEXTAREA" && !ev.metaKey && !ev.ctrlKey) {
        ev.preventDefault();
        input.focus();
      }
    });
  }

  /* ------------------------------------------------------------- repository */
  function repoFacts() {
    var facts = document.querySelector(".bl-repo-facts");
    if (!facts || location.protocol === "file:") return;
    var key = "bl-repo-facts";
    function show(d) {
      facts.innerHTML = "";
      [["tag", d.v], ["star", d.s], ["fork", d.f]].forEach(function (f) {
        if (f[1] == null) return;
        facts.appendChild(el("span", "bl-fact bl-fact-" + f[0], String(f[1])));
      });
    }
    try {
      var cached = JSON.parse(sessionStorage.getItem(key) || "null");
      if (cached) return show(cached);
    } catch (e) { /* storage blocked: fetch every time */ }
    fetch("https://api.github.com/repos/jonasdelacour/BatchLAS").then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        if (!j) return;
        var d = { v: facts.dataset.version ? "v" + facts.dataset.version : null, s: j.stargazers_count, f: j.forks_count };
        try { sessionStorage.setItem(key, JSON.stringify(d)); } catch (e) { /* ignore */ }
        show(d);
      }).catch(function () { /* offline: keep the version only */ });
  }

  function init() {
    buildNav();
    buildToc();
    buildSearch();
    repoFacts();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
