/* MIRA — i18n. Full translations: English, Hindi, Marathi.
 * Core UI (nav, buttons, labels, section titles) additionally for major
 * Indian languages; long-form content falls back to English rather than
 * shipping sloppy machine translation. Language persists in localStorage.
 */
(function () {
  "use strict";

  var CORE_LANGS = ["bn", "ta", "te", "gu", "pa", "kn", "ml", "or"];

  var I18N = {
    en: {
      "nav.research": "Research", "nav.architecture": "Architecture",
      "nav.science": "Science", "nav.console": "Open Console →",
      "nav.overview": "Overview", "nav.chat": "Chat", "nav.mandala": "Mandala",
      "nav.documents": "Documents", "nav.websearch": "Web Search",
      "nav.lab": "Research Lab", "nav.benchmarks": "Benchmarks",
      "hero.eyebrow": "Local-first memory research",
      "hero.h1": "Memory, arranged<br>like a <em>mandala</em>.",
      "hero.sub": "MIRA tests a formal hypothesis: can a <strong>radial, hierarchical, graph-based memory topology</strong>, using rings and sectors instead of flat vectors, improve how machines retrieve and reason? The system is built to prove it wrong as easily as right.",
      "hero.cta1": "Launch the console", "hero.cta2": "Read the research question",
      "hero.stats.components": "Retrieval components", "hero.stats.ablations": "Ablation sets",
      "hero.stats.baselines": "Baselines", "hero.stats.cloud": "Cloud calls (default)",
      "honesty.body": "<strong>What MIRA is not.</strong> MIRA is an experimental architecture inspired by the organizational and visual principles of mandalas/yantras. It does <strong>not</strong> claim that historical mandalas were neural networks or that ancient traditions contained modern AI. Whether radial organization helps is an <strong>empirical question</strong>: negative results are reported, not hidden.",
      "research.h2": "Does radial topology earn its place?",
      "research.lede": "When model, context budget, dataset and hardware are held constant, does arranging memory in concentric rings around a core, measured by a formal radial distance, retrieve better evidence for multi-hop questions than flat vector search?",
      "research.card1": "The formula under test", "research.card2": "The honest baselines",
      "research.card3": "What counts as a win",
      "architecture.h2": "From PDF to audited answer",
      "science.h2": "Built to be refuted",
      "bench.h2": "Evaluated on real multi-hop data",
      "bench.lede": "MuSiQue (Trivedi et al., 2021): answerable 2-hop questions over a shared distractor corpus. MIRA is compared against flat-vector retrieval with an identical answer stage; per-question paired bootstrap and Wilcoxon tests back every comparison.",
      "chat.placeholder": "Ask a multi-hop question…",
      "chat.ask": "Ask", "chat.web": "Allow live web",
      "chat.listen": "Speak your question", "chat.speak": "Read the answer aloud",
      "chat.stop": "Stop", "chat.mic.unsupported": "Voice input needs Chrome or Edge",
      "chat.listening": "Listening…", "lang.label": "Language",
      "footer.built": "Created by Aryan Chavan. Local-first research prototype. Runs entirely on your machine.",
      "footer.console": "Console", "footer.api": "API docs", "footer.source": "Source",
      "affect.title": "Simulated affect state",
      "affect.note": "Simulated algorithmic state, not consciousness.",
      "loading": "Loading system…"
    },
    hi: {
      "nav.research": "शोध", "nav.architecture": "संरचना",
      "nav.science": "विज्ञान", "nav.console": "कंसोल खोलें →",
      "nav.overview": "अवलोकन", "nav.chat": "संवाद", "nav.mandala": "मंडल",
      "nav.documents": "दस्तावेज़", "nav.websearch": "वेब खोज",
      "nav.lab": "शोध प्रयोगशाला", "nav.benchmarks": "बेंचमार्क",
      "hero.eyebrow": "स्थानीय-प्रथम स्मृति शोध",
      "hero.h1": "स्मृति, व्यवस्थित<br>एक <em>मंडल</em> की तरह।",
      "hero.sub": "MIRA एक औपचारिक परिकल्पना जाँचता है: क्या फ़्लैट वेक्टर के बजाय वलय और क्षेत्रों वाली <strong>त्रिज्य, पदानुक्रमित, ग्राफ़-आधारित स्मृति संरचना</strong> मशीनों को बेहतर पुनर्प्राप्ति और तर्क करने देती है? यह तंत्र ग़लत साबित होने को उतनी ही आसानी से स्वीकार करता है जितनी सही होने को।",
      "hero.cta1": "कंसोल खोलें", "hero.cta2": "शोध प्रश्न पढ़ें",
      "hero.stats.components": "पुनर्प्राप्ति घटक", "hero.stats.ablations": "निरसन सेट",
      "hero.stats.baselines": "आधारभूत तंत्र", "hero.stats.cloud": "क्लाउड कॉल (डिफ़ॉल्ट)",
      "honesty.body": "<strong>MIRA क्या नहीं है।</strong> MIRA मंडल/यंत्र की संगठनात्मक और दृश्य सिद्धांतों से प्रेरित एक प्रायोगिक संरचना है। यह <strong>दावा नहीं करता</strong> कि ऐतिहासिक मंडल तंत्रिका-जाल थे या प्राचीन परंपराओं में आधुनिक AI था। क्या त्रिज्य संगठन मदद करता है, यह एक <strong>प्रायोगिक प्रश्न</strong> है: नकारात्मक परिणाम भी प्रकाशित होते हैं, छिपाए नहीं जाते।",
      "research.h2": "क्या त्रिज्य संरचना अपनी जगह साबित करती है?",
      "research.lede": "जब मॉडल, संदर्भ-बजट, डेटासेट और हार्डवेयर स्थिर रहते हैं, तब क्या स्मृति को केंद्र के चारों ओर संकेंद्रित वलयों में — औपचारिक त्रिज्य दूरी से मापी हुई — व्यवस्थित करना फ़्लैट वेक्टर खोज से बेहतर बहु-चरण (multi-hop) साक्ष्य पुनर्प्राप्त करता है?",
      "research.card1": "परीक्षित सूत्र", "research.card2": "ईमानदार आधारभूत तंत्र",
      "research.card3": "जीत का मानक",
      "architecture.h2": "PDF से लेकर ऑडिट-योग्य उत्तर तक",
      "science.h2": "खंडन के लिए बनाया गया",
      "bench.h2": "वास्तविक बहु-चरण डेटा पर मूल्यांकन",
      "bench.lede": "MuSiQue (त्रिवेदी एवं अन्य, 2021): साझा विकर्षक (distractor) कॉर्पस पर उत्तर-योग्य 2-चरण प्रश्न। MIRA की तुलना समान उत्तर-चरण वाले फ़्लैट-वेक्टर पुनर्प्राप्ति से की जाती है; हर तुलना प्रति-प्रश्न जोड़ीदार बूटस्ट्रैप और विलकॉक्सन परीक्षण से समर्थित है।",
      "chat.placeholder": "बहु-चरण प्रश्न पूछें…",
      "chat.ask": "पूछें", "chat.web": "सीधे वेब की अनुमति दें",
      "chat.listen": "अपना प्रश्न बोलें", "chat.speak": "उत्तर पढ़कर सुनाएँ",
      "chat.stop": "रोकें", "chat.mic.unsupported": "ध्वनि इनपुट के लिए Chrome या Edge चाहिए",
      "chat.listening": "सुन रहा है…", "lang.label": "भाषा",
      "footer.built": "आर्यन चव्हाण द्वारा निर्मित। स्थानीय-प्रथम शोध प्रोटोटाइप। पूरी तरह आपकी मशीन पर चलता है।",
      "footer.console": "कंसोल", "footer.api": "API दस्तावेज़", "footer.source": "स्रोत",
      "affect.title": "अनुकरित भाव अवस्था",
      "affect.note": "अनुकरित कलन-आधारित अवस्था, चेतना नहीं।",
      "loading": "तंत्र लोड हो रहा है…"
    },
    mr: {
      "nav.research": "संशोधन", "nav.architecture": "रचना",
      "nav.science": "विज्ञान", "nav.console": "कन्सोल उघडा →",
      "nav.overview": "आढावा", "nav.chat": "संवाद", "nav.mandala": "मंडल",
      "nav.documents": "दस्तऐवज", "nav.websearch": "वेब शोध",
      "nav.lab": "संशोधन प्रयोगशाळा", "nav.benchmarks": "बेंचमार्क",
      "hero.eyebrow": "स्थानिक-प्रथम स्मृती संशोधन",
      "hero.h1": "स्मृती, <em>मंडला</em>सारखी<br>मांडलेली.",
      "hero.sub": "MIRA एक औपचारिक गृहितक तपासते: सपाट व्हेक्टरच्या जागी वलय आणि क्षेत्रांची <strong>त्रिज्य, क्रमोच्च, ग्राफ-आधारित स्मृती रचना</strong> यंत्रांना चांगली पुनर्प्राप्ती आणि तर्क करू देते का? ही प्रणाली चुकीची ठरवणे जितके सोपे आहे तितकेच बरोबर ठरवणेही सोपे आहे.",
      "hero.cta1": "कन्सोल उघडा", "hero.cta2": "संशोधन प्रश्न वाचा",
      "hero.stats.components": "पुनर्प्राप्ती घटक", "hero.stats.ablations": "निरसन संच",
      "hero.stats.baselines": "आधारभूत प्रणाली", "hero.stats.cloud": "क्लाउड कॉल (मूलभूत)",
      "honesty.body": "<strong>MIRA काय नाही.</strong> MIRA हा मंडल/यंत्रांच्या संघटनात्मक आणि दृक् तत्त्वांवरून प्रेरित एक प्रायोगिक रचना आहे. हा <strong>दावा करत नाही</strong> की ऐतिहासिक मंडले तंत्रिका-जाल होती किंवा प्राचीन परंपरांमध्ये आधुनिक AI होते. त्रिज्य संघटना उपयोगी ठरते का हा एक <strong>प्रायोगिक प्रश्न</strong> आहे: नकारात्मक निष्कर्ष छापले जातात, लपवले जात नाहीत.",
      "research.h2": "त्रिज्य रचना आपली जागा सिद्ध करते का?",
      "research.lede": "मॉडेल, संदर्भ-अंशतः, डेटासेट आणि हार्डवेअर स्थिर असताना, स्मृतीला केंद्राभोवती संकेंद्रित वलयांत — औपचारिक त्रिज्य अंतराने मोजलेली — मांडणे, सपाट व्हेक्टर शोधापेक्षा बहु-टप्पू (multi-hop) पुराव्याची चांगली पुनर्प्राप्ती करते का?",
      "research.card1": "चाचणीखालील सूत्र", "research.card2": "प्रामाणिक आधारभूत प्रणाली",
      "research.card3": "विजयाची मानके",
      "architecture.h2": "PDF पासून ऑडिट-योग्य उत्तरापर्यंत",
      "science.h2": "खंडनासाठी बांधलेले",
      "bench.h2": "प्रत्यक्ष बहु-टप्पू डेटावर मूल्यांकन",
      "bench.lede": "MuSiQue (त्रिवेदी इ. 2021): सामायिक विकर्षक कॉर्पसवर उत्तरदायी 2-टप्पू प्रश्न. MIRA ची तुलना समान उत्तर-टप्प्याच्या सपाट-व्हेक्टर पुनर्प्राप्तीशी होते; प्रत्येक तुलना प्रति-प्रश्न जोडीदार बूटस्ट्रॅप आणि विलकॉक्सन चाचणींनी समर्थित आहे.",
      "chat.placeholder": "बहु-टप्पू प्रश्न विचारा…",
      "chat.ask": "विचारा", "chat.web": "प्रत्यक्ष वेबला परवानगी द्या",
      "chat.listen": "आपला प्रश्न बोला", "chat.speak": "उत्तर ऐकवा",
      "chat.stop": "थांबा", "chat.mic.unsupported": "आवाज इनपुटसाठी Chrome किंवा Edge लागतो",
      "chat.listening": "ऐकत आहे…", "lang.label": "भाषा",
      "footer.built": "आर्यन चव्हाण यांनी निर्मित. स्थानिक-प्रथम संशोधन प्रोटोटाइप. संपूर्णपणे तुमच्या मशीनवर चालते.",
      "footer.console": "कन्सोल", "footer.api": "API दस्तऐवज", "footer.source": "स्रोत",
      "affect.title": "अनुकरित भाव अवस्था",
      "affect.note": "अनुकरित गणकीय अवस्था, चेतना नाही.",
      "loading": "प्रणाली लोड होत आहे…"
    }
  };

  /* Core-UI coverage for other major Indian languages (nav/buttons/labels). */
  var CORE = {
    bn: {"nav.research": "গবেষণা", "nav.architecture": "স্থাপত্য", "nav.science": "বিজ্ঞান",
         "nav.console": "কনসোল খুলুন →", "nav.overview": "সংক্ষিপ্ত চিত্র", "nav.chat": "সংলাপ",
         "nav.mandala": "মণ্ডল", "nav.documents": "নথি", "nav.websearch": "ওয়েব অনুসন্ধান",
         "nav.lab": "গবেষণা পরীক্ষাগার", "nav.benchmarks": "বেঞ্চমার্ক",
         "chat.ask": "জিজ্ঞাসা করুন", "chat.placeholder": "একটি বহু-ধাপ প্রশ্ন করুন…",
         "chat.web": "সরাসরি ওয়েব অনুমোদন", "lang.label": "ভাষা", "loading": "সিস্টেম লোড হচ্ছে…"},
    ta: {"nav.research": "ஆய்வு", "nav.architecture": "கட்டமைப்பு", "nav.science": "அறிவியல்",
         "nav.console": "கன்சோலைத் திற →", "nav.overview": "மேலோட்டம்", "nav.chat": "உரையாடல்",
         "nav.mandala": "மண்டலம்", "nav.documents": "ஆவணங்கள்", "nav.websearch": "இணைய தேடல்",
         "nav.lab": "ஆய்வுக் கூடம்", "nav.benchmarks": "தரநிர்ணயம்",
         "chat.ask": "கேளுங்கள்", "chat.placeholder": "பல-படி கேள்வி கேளுங்கள்…",
         "chat.web": "நேரடி இணைய அனுமதி", "lang.label": "மொழி", "loading": "அமைப்பு ஏற்றப்படுகிறது…"},
    te: {"nav.research": "పరిశోధన", "nav.architecture": "నిర్మాణం", "nav.science": "విజ్ఞానం",
         "nav.console": "కన్సోల్ తెరువు →", "nav.overview": "సూక్ష్మదృష్టి", "nav.chat": "సంభాషణ",
         "nav.mandala": "మండలం", "nav.documents": "పత్రాలు", "nav.websearch": "వెబ్ శోధన",
         "nav.lab": "పరిశోధన ప్రయోగశాల", "nav.benchmarks": "బెంచ్‌మార్క్‌లు",
         "chat.ask": "అడగండి", "chat.placeholder": "బహు-దశ ప్రశ్న అడగండి…",
         "chat.web": "ప్రత్యక్ష వెబ్ అనుమతి", "lang.label": "భాష", "loading": "వ్యవస్థ లోడ్ అవుతోంది…"},
    gu: {"nav.research": "સંશોધન", "nav.architecture": "માળખું", "nav.science": "વિજ્ઞાન",
         "nav.console": "કન્સોલ ખોલો →", "nav.overview": "ઝાંખી", "nav.chat": "સંવાદ",
         "nav.mandala": "મંડળ", "nav.documents": "દસ્તાવેજો", "nav.websearch": "વેબ શોધ",
         "nav.lab": "સંશોધન પ્રયોગશાળા", "nav.benchmarks": "બેન્ચમાર્ક",
         "chat.ask": "પૂછો", "chat.placeholder": "બહુ-તબક્કાનો પ્રશ્ન પૂછો…",
         "chat.web": "સીધા વેબની મંજૂરી", "lang.label": "ભાષા", "loading": "સિસ્ટમ લોડ થાય છે…"},
    pa: {"nav.research": "ਖੋਜ", "nav.architecture": "ਢਾਂਚਾ", "nav.science": "ਵਿਗਿਆਨ",
         "nav.console": "ਕਨਸੋਲ ਖੋਲ੍ਹੋ →", "nav.overview": "ਝਲਕ", "nav.chat": "ਗੱਲਬਾਤ",
         "nav.mandala": "ਮੰਡਲ", "nav.documents": "ਦਸਤਾਵੇਜ਼", "nav.websearch": "ਵੈੱਬ ਖੋਜ",
         "nav.lab": "ਖੋਜ ਪ੍ਰਯੋਗਸ਼ਾਲਾ", "nav.benchmarks": "ਬੈਂਚਮਾਰਕ",
         "chat.ask": "ਪੁੱਛੋ", "chat.placeholder": "ਬਹੁ-ਪੜਾਅ ਸਵਾਲ ਪੁੱਛੋ…",
         "chat.web": "ਸਿੱਧੀ ਵੈੱਬ ਇਜਾਜ਼ਤ", "lang.label": "ਭਾਸ਼ਾ", "loading": "ਸਿਸਟਮ ਲੋਡ ਹੋ ਰਿਹਾ ਹੈ…"},
    kn: {"nav.research": "ಸಂಶೋಧನೆ", "nav.architecture": "ರಚನೆ", "nav.science": "ವಿಜ್ಞಾನ",
         "nav.console": "ಕನ್ಸೋಲ್ ತೆರೆಯಿರಿ →", "nav.overview": "ಅವಲೋಕನ", "nav.chat": "ಸಂಭಾಷಣೆ",
         "nav.mandala": "ಮಂಡಲ", "nav.documents": "ದಸ್ತಾವೇಜುಗಳು", "nav.websearch": "ವೆಬ್ ಹುಡುಕಾಟ",
         "nav.lab": "ಸಂಶೋಧನಾ ಪ್ರಯೋಗಾಲಯ", "nav.benchmarks": "ಮಾನದಂಡಗಳು",
         "chat.ask": "ಕೇಳಿ", "chat.placeholder": "ಬಹು-ಹಂತದ ಪ್ರಶ್ನೆ ಕೇಳಿ…",
         "chat.web": "ನೇರ ವೆಬ್ ಅನುಮತಿ", "lang.label": "ಭಾಷೆ", "loading": "ವ್ಯವಸ್ಥೆ ಲೋಡ್ ಆಗುತ್ತಿದೆ…"},
    ml: {"nav.research": "ഗവേഷണം", "nav.architecture": "ഘടന", "nav.science": "ശാസ്ത്രം",
         "nav.console": "കൺസോൾ തുറക്കുക →", "nav.overview": "അവലോകനം", "nav.chat": "സംഭാഷണം",
         "nav.mandala": "മണ്ഡലം", "nav.documents": "രേഖകൾ", "nav.websearch": "വെബ് തിരച്ചിൽ",
         "nav.lab": "ഗവേഷണ ലാബ്", "nav.benchmarks": "ബഞ്ച്മാർക്കുകൾ",
         "chat.ask": "ചോദിക്കുക", "chat.placeholder": "ഒരു ബഹുഘട്ട ചോദ്യം ചോദിക്കുക…",
         "chat.web": "നേരിട്ടുള്ള വെബ് അനുവദിക്കുക", "lang.label": "ഭാഷ", "loading": "സിസ്റ്റം ലോഡ് ചെയ്യുന്നു…"},
    or: {"nav.research": "ଗବେଷଣା", "nav.architecture": "ସ୍ଥାପତ୍ୟ", "nav.science": "ବିଜ୍ଞାନ",
         "nav.console": "କନସୋଲ ଖୋଲନ୍ତୁ →", "nav.overview": "ସଂକ୍ଷିପ୍ତ ଦୃଶ୍ୟ", "nav.chat": "ସଂଳାପ",
         "nav.mandala": "ମଣ୍ଡଳ", "nav.documents": "ଦଲିଲ", "nav.websearch": "ୱେବ୍ ଖୋଜ",
         "nav.lab": "ଗବେଷଣା ପରୀକ୍ଷାଗାର", "nav.benchmarks": "ବେଞ୍ଚମାର୍କ",
         "chat.ask": "ପଚାରନ୍ତୁ", "chat.placeholder": "ଏକ ବହୁ-ପାଦ ପ୍ରଶ୍ନ ପଚାରନ୍ତୁ…",
         "chat.web": "ସିଧା ୱେବ ଅନୁମତି", "lang.label": "ଭାଷା", "loading": "ସିଷ୍ଟମ୍ ଲୋଡ୍ ହେଉଛି…"}
  };
  Object.keys(CORE).forEach(function (code) { I18N[code] = CORE[code]; });

  var LANGS = [["en", "English"], ["hi", "हिन्दी"], ["mr", "मराठी"], ["bn", "বাংলা"],
               ["ta", "தமிழ்"], ["te", "తెలుగు"], ["gu", "ગુજરાતી"], ["pa", "ਪੰਜਾਬੀ"],
               ["kn", "ಕನ್ನಡ"], ["ml", "മലയാളം"], ["or", "ଓଡ଼ିଆ"]];
  var BCP47 = { en: "en", hi: "hi-IN", mr: "mr-IN", bn: "bn-IN", ta: "ta-IN",
                te: "te-IN", gu: "gu-IN", pa: "pa-IN", kn: "kn-IN", ml: "ml-IN", or: "or-IN" };

  var current = "en";
  try { current = localStorage.getItem("mira_lang") || "en"; } catch (e) { /* private mode */ }
  if (!I18N[current]) current = "en";

  function t(key) {
    return (I18N[current] && I18N[current][key]) || I18N.en[key] || key;
  }

  function apply() {
    document.documentElement.lang = BCP47[current] || current;
    document.querySelectorAll("[data-i18n]").forEach(function (el) {
      var key = el.getAttribute("data-i18n");
      var val = t(key);
      if (val !== key) el.innerHTML = val;
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach(function (el) {
      var val = t(el.getAttribute("data-i18n-placeholder"));
      if (val) el.placeholder = val;
    });
    document.querySelectorAll("[data-i18n-aria]").forEach(function (el) {
      var val = t(el.getAttribute("data-i18n-aria"));
      if (val) el.setAttribute("aria-label", val);
    });
    document.querySelectorAll("[data-i18n-title]").forEach(function (el) {
      var val = t(el.getAttribute("data-i18n-title"));
      if (val) el.title = val;
    });
    document.dispatchEvent(new CustomEvent("mira:langchange", { detail: { lang: current } }));
  }

  function setLang(code) {
    if (!I18N[code]) return;
    current = code;
    try { localStorage.setItem("mira_lang", code); } catch (e) { /* ignore */ }
    apply();
  }

  function switcherHTML(selectedId) {
    var opts = LANGS.map(function (l) {
      return '<option value="' + l[0] + '"' + (l[0] === selectedId ? " selected" : "") +
             ">" + l[1] + "</option>";
    }).join("");
    return '<label class="lang-switch"><span class="visually-hidden">' +
           t("lang.label") + '</span><select data-i18n-aria="lang.label" class="lang-select">' +
           opts + "</select></label>";
  }

  function mountSwitchers() {
    document.querySelectorAll(".lang-mount").forEach(function (mount) {
      mount.innerHTML = switcherHTML(current);
      var sel = mount.querySelector("select");
      sel.addEventListener("change", function () { setLang(sel.value); });
    });
  }

  window.MIRAI18N = { t: t, setLang: setLang, get lang() { return current; },
                      bcp47: function (code) { return BCP47[code || current] || "en"; },
                      langs: LANGS };

  function boot() {
    mountSwitchers();
    apply();
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
