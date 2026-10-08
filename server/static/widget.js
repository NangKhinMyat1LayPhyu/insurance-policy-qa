(function () {
  "use strict";

  var scriptTag = document.currentScript;
  var API_URL = (scriptTag && scriptTag.getAttribute("data-api-url")) || "http://localhost:8000";

  var SESSION_KEY = "kasturi_widget_session_id";

  function getSessionId() {
    try {
      var id = localStorage.getItem(SESSION_KEY);
      if (!id) {
        id = crypto.randomUUID();
        localStorage.setItem(SESSION_KEY, id);
      }
      return id;
    } catch (e) {
      return crypto.randomUUID();
    }
  }
  var sessionId = getSessionId();

  var css =
    "#kasturi-widget-bubble{position:fixed;bottom:20px;right:20px;width:56px;height:56px;" +
    "border-radius:50%;background:#0f4c81;color:#fff;display:flex;align-items:center;" +
    "justify-content:center;cursor:pointer;box-shadow:0 4px 12px rgba(0,0,0,.2);font-size:24px;" +
    "z-index:999999;border:none;}" +
    "#kasturi-widget-panel{position:fixed;bottom:88px;right:20px;width:340px;max-width:90vw;" +
    "height:460px;max-height:70vh;background:#fff;border-radius:12px;" +
    "box-shadow:0 8px 30px rgba(0,0,0,.25);display:none;flex-direction:column;overflow:hidden;" +
    "z-index:999999;font-family:system-ui,-apple-system,sans-serif;}" +
    "#kasturi-widget-panel.open{display:flex;}" +
    "#kasturi-widget-header{background:#0f4c81;color:#fff;padding:12px 16px;font-weight:600;font-size:14px;}" +
    "#kasturi-widget-messages{flex:1;overflow-y:auto;padding:12px;font-size:13px;background:#fafbfc;}" +
    ".kw-msg{margin-bottom:10px;padding:8px 12px;border-radius:10px;max-width:85%;" +
    "white-space:pre-wrap;line-height:1.4;}" +
    ".kw-msg.user{background:#0f4c81;color:#fff;margin-left:auto;}" +
    ".kw-msg.bot{background:#f0f2f5;color:#1a1a1a;}" +
    ".kw-msg.bot.loading{color:#888;font-style:italic;}" +
    ".kw-sources{font-size:11px;color:#666;margin-top:4px;}" +
    ".kw-handoff{font-size:12px;color:#444;margin-top:6px;padding-top:6px;border-top:1px dashed #ddd;}" +
    ".kw-cache-badge{display:inline-block;font-size:10px;background:#e6f4ea;color:#1e7e34;" +
    "border-radius:4px;padding:1px 6px;margin-bottom:4px;}" +
    "#kasturi-widget-form{display:flex;border-top:1px solid #e5e5e5;}" +
    "#kasturi-widget-input{flex:1;border:none;padding:10px 12px;font-size:13px;outline:none;}" +
    "#kasturi-widget-send{border:none;background:#0f4c81;color:#fff;padding:0 16px;cursor:pointer;font-size:13px;}" +
    "#kasturi-widget-send:disabled{opacity:.5;cursor:default;}";
  var styleTag = document.createElement("style");
  styleTag.textContent = css;
  document.head.appendChild(styleTag);

  var bubble = document.createElement("button");
  bubble.id = "kasturi-widget-bubble";
  bubble.setAttribute("aria-label", "Open support chat");
  bubble.textContent = "💬";
  document.body.appendChild(bubble);

  var panel = document.createElement("div");
  panel.id = "kasturi-widget-panel";
  panel.innerHTML =
    '<div id="kasturi-widget-header">Kasturi HomeCare &amp; Health -- Ask a question</div>' +
    '<div id="kasturi-widget-messages"></div>' +
    '<form id="kasturi-widget-form">' +
    '<input id="kasturi-widget-input" type="text" placeholder="Ask about your policy..." autocomplete="off" />' +
    '<button id="kasturi-widget-send" type="submit">Send</button>' +
    "</form>";
  document.body.appendChild(panel);

  var messagesEl = panel.querySelector("#kasturi-widget-messages");
  var formEl = panel.querySelector("#kasturi-widget-form");
  var inputEl = panel.querySelector("#kasturi-widget-input");
  var sendEl = panel.querySelector("#kasturi-widget-send");

  bubble.addEventListener("click", function () {
    panel.classList.toggle("open");
    if (panel.classList.contains("open")) inputEl.focus();
  });

  function escapeHtml(s) {
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function addTextMessage(role, text) {
    var div = document.createElement("div");
    div.className = "kw-msg " + role;
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  function addBotResult(data) {
    var div = document.createElement("div");
    div.className = "kw-msg bot";
    var html = "";
    if (data.cache_hit) {
      html += '<span class="kw-cache-badge">instant answer</span><br/>';
    }
    html += escapeHtml(data.answer).replace(/\n/g, "<br/>");
    if (data.sources && data.sources.length) {
      html +=
        '<div class="kw-sources">Sources: ' +
        data.sources
          .map(function (s) {
            return escapeHtml(s.document) + " p." + s.page;
          })
          .join(", ") +
        "</div>";
    }
    if (data.needs_human) {
      html += '<div class="kw-handoff">I\'ve flagged this conversation for our support team to review.</div>';
    }
    div.innerHTML = html;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  addTextMessage("bot", "Hi! Ask me anything about your Kasturi HomeCare & Health Plan policy.");

  formEl.addEventListener("submit", function (e) {
    e.preventDefault();
    var text = inputEl.value.trim();
    if (!text) return;

    addTextMessage("user", text);
    inputEl.value = "";
    inputEl.disabled = true;
    sendEl.disabled = true;
    var loadingEl = addTextMessage("bot loading", "Thinking...");

    fetch(API_URL + "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    })
      .then(function (res) {
        if (!res.ok) throw new Error("Request failed: " + res.status);
        return res.json();
      })
      .then(function (data) {
        loadingEl.remove();
        addBotResult(data);
      })
      .catch(function (err) {
        loadingEl.remove();
        addTextMessage("bot", "Sorry, something went wrong reaching the support system. Please try again.");
        console.error("Kasturi widget error:", err);
      })
      .finally(function () {
        inputEl.disabled = false;
        sendEl.disabled = false;
        inputEl.focus();
      });
  });
})();
