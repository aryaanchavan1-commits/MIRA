/* MIRA — live voice: natural spoken interaction.
 * Web Speech API (built into Chrome/Edge — no new dependency, works offline
 * for TTS; STT uses the browser's on-device/streaming recognizer).
 * Language follows the UI language; graceful no-op where unsupported.
 */
(function () {
  "use strict";

  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  var supported = { stt: !!SR, tts: "speechSynthesis" in window };
  var recognizing = false;
  var recognition = null;
  var onResult = null;
  var onEnd = null;

  function lang() {
    return (window.MIRAI18N && window.MIRAI18N.bcp47()) || "en";
  }

  function startListening(handlers) {
    if (!SR) return false;
    onResult = handlers && handlers.onResult;
    onEnd = handlers && handlers.onEnd;
    if (recognizing) return true;
    recognition = new SR();
    recognition.lang = lang();
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = function (e) {
      var text = "";
      for (var i = 0; i < e.results.length; i++) {
        text += e.results[i][0].transcript;
      }
      if (onResult) onResult(text, e.results[e.results.length - 1].isFinal);
    };
    recognition.onerror = function (e) {
      if (onEnd) onEnd(e.error === "no-speech" ? null : e.error);
      recognizing = false;
    };
    recognition.onend = function () {
      recognizing = false;
      if (onEnd) onEnd(null);
    };
    try {
      recognition.start();
      recognizing = true;
      return true;
    } catch (e) {
      return false;
    }
  }

  function stopListening() {
    if (recognition && recognizing) {
      try { recognition.stop(); } catch (e) { /* already stopped */ }
    }
    recognizing = false;
  }

  function isListening() { return recognizing; }

  /* Speak a text; strips citation markers like [3] for a natural read. */
  function speak(text) {
    if (!supported.tts || !text) return;
    stopSpeaking();
    var clean = String(text).replace(/\[\d+\]/g, "").replace(/\s+/g, " ").trim();
    var u = new SpeechSynthesisUtterance(clean);
    u.lang = lang();
    u.rate = 1.02;
    u.pitch = 1.0;
    var voices = window.speechSynthesis.getVoices() || [];
    var match = voices.find(function (v) { return v.lang === u.lang; }) ||
                voices.find(function (v) { return v.lang && v.lang.indexOf(u.lang.split("-")[0]) === 0; });
    if (match) u.voice = match;
    window.speechSynthesis.speak(u);
  }

  function stopSpeaking() {
    if (supported.tts) window.speechSynthesis.cancel();
  }

  function isSpeaking() {
    return supported.tts && window.speechSynthesis.speaking;
  }

  window.MIRAVoice = { supported: supported, startListening: startListening,
                       stopListening: stopListening, isListening: isListening,
                       speak: speak, stopSpeaking: stopSpeaking, isSpeaking: isSpeaking };
})();
