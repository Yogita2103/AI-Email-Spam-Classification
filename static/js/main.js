// main.js - talks to the Flask backend and updates the page.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const textArea = $("emailText");
  const analyzeBtn = $("analyzeBtn");
  const errorMsg = $("errorMsg");

  const SAMPLES = {
    spam:
      "Subject: CONGRATULATIONS!!! You have WON $1,000,000\n\n" +
      "Dear lucky winner, you have been selected to receive a FREE cash prize. " +
      "Click here http://claim-your-prize.biz now to claim your money. " +
      "Act fast, this limited offer expires today! Send your bank details to receive the payment.",
    ham:
      "Subject: Project review meeting tomorrow\n\n" +
      "Hi Priya,\n\nI have attached the agenda for tomorrow's project review. " +
      "Please read the report before the meeting at 10am and send me any questions. " +
      "We will discuss the schedule for the next sprint.\n\nThanks,\nRahul",
  };

  function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.hidden = false;
  }

  function updateCounter() {
    const max = textArea.getAttribute("maxlength");
    $("counter").textContent = textArea.value.length + " / " + max;
  }

  function showVerdict(result) {
    const stamp = $("stamp");
    stamp.textContent = result.is_spam ? "Spam" : "Not spam";
    stamp.className = "stamp " + (result.is_spam ? "is-spam" : "is-clean");
    void stamp.offsetWidth; // restart the stamp animation
    stamp.classList.add("thump");
    $("spamProb").textContent = result.spam_probability.toFixed(1) + "%";
    $("confidence").textContent = result.confidence.toFixed(1) + "%";
    $("meterFill").style.width = result.spam_probability + "%";
    $("verdictEmpty").hidden = true;
    $("verdictResult").hidden = false;
  }

  async function analyze() {
    errorMsg.hidden = true;
    const text = textArea.value.trim();
    if (!text) {
      showError("Paste an email above, then press Check this email.");
      return;
    }
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Checking...";
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed.");
      showVerdict(data);
    } catch (err) {
      showError(err.message || "Could not reach the server. Try again.");
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Check this email";
    }
  }

  function pct(v) {
    return (v * 100).toFixed(1) + "%";
  }

  async function loadMetrics() {
    try {
      const res = await fetch("/metrics");
      const m = await res.json();
      if (!res.ok) throw new Error(m.error || "Metrics unavailable.");
      $("mAccuracy").textContent = pct(m.accuracy);
      $("mPrecision").textContent = pct(m.precision);
      $("mRecall").textContent = pct(m.recall);
      $("mF1").textContent = pct(m.f1_score);
      const c = m.confusion_matrix;
      $("cTN").textContent = c.true_negative;
      $("cFP").textContent = c.false_positive;
      $("cFN").textContent = c.false_negative;
      $("cTP").textContent = c.true_positive;

      const facts = $("facts");
      facts.innerHTML = "";
      [
        ["Emails in dataset", m.total_emails],
        ["Spam", m.spam_emails],
        ["Not spam", m.ham_emails],
        ["Trained on", m.train_size],
        ["Tested on", m.test_size],
        ["Model", m.model],
      ].forEach(function (pair) {
        const span = document.createElement("span");
        const strong = document.createElement("strong");
        strong.textContent = pair[1];
        span.append(pair[0] + ": ", strong);
        facts.appendChild(span);
      });

      if (m.top_spam_words && m.top_spam_words.length) {
        const box = $("words");
        box.innerHTML = "";
        m.top_spam_words.forEach(function (w) {
          const s = document.createElement("span");
          s.textContent = w;
          box.appendChild(s);
        });
        $("wordsBlock").hidden = false;
      }
    } catch (err) {
      const box = $("metricsError");
      box.textContent = err.message;
      box.hidden = false;
    }
  }

  analyzeBtn.addEventListener("click", analyze);
  $("clearBtn").addEventListener("click", function () {
    textArea.value = "";
    errorMsg.hidden = true;
    $("verdictResult").hidden = true;
    $("verdictEmpty").hidden = false;
    updateCounter();
    textArea.focus();
  });
  textArea.addEventListener("input", updateCounter);
  textArea.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
  });
  document.querySelectorAll("[data-sample]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      textArea.value = SAMPLES[btn.dataset.sample];
      updateCounter();
    });
  });

  updateCounter();
  loadMetrics();
})();
