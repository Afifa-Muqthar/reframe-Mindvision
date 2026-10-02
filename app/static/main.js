const form = document.getElementById("worry-form");
const textInput = document.getElementById("text-input");
const audioInput = document.getElementById("audio-input");
const loadingSection = document.getElementById("loading-section");
const resultSection = document.getElementById("result-section");
const crisisSection = document.getElementById("crisis-section");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  resultSection.classList.add("hidden");
  crisisSection.classList.add("hidden");
  loadingSection.classList.remove("hidden");

  const formData = new FormData();
  formData.append("text", textInput.value);
  if (audioInput.files[0]) {
    formData.append("audio", audioInput.files[0]);
  }

  try {
    const res = await fetch("/process", { method: "POST", body: formData });
    const data = await res.json();
    loadingSection.classList.add("hidden");

    if (data.error) {
      alert(data.error);
      return;
    }

    if (data.crisis) {
      crisisSection.classList.remove("hidden");
      crisisSection.querySelector(".crisis-box").textContent = data.message;
      return;
    }

    // Populate Psychological Reasoning Layer
    const reasoning = data.reasoning || {};
    document.getElementById("reasoning-happening").textContent =
      reasoning.what_may_be_happening || data.pattern || "Situation reflection";
    document.getElementById("reasoning-control").textContent =
      reasoning.what_you_can_control || "Focus on personal boundaries, support, and agency.";
    const uncontrolEl = document.getElementById("reasoning-uncontrollable");
    if (uncontrolEl) {
      uncontrolEl.textContent =
        reasoning.what_is_not_controllable || "Past events and external actions of others.";
    }
    document.getElementById("reasoning-reframe").textContent =
      `“${reasoning.reframe || ""}”`;
    document.getElementById("reasoning-action").textContent =
      reasoning.next_step || "Take one manageable constructive step today.";

    // Badge for external threat vs internal pattern
    const badge = document.getElementById("threat-badge");
    if (data.is_external_threat) {
      badge.textContent = "Trauma-Informed Safety Focus";
      badge.className = "badge badge-threat";
    } else {
      badge.textContent = "Cognitive Coping Pattern";
      badge.className = "badge badge-cognitive";
    }

    document.getElementById("citation").textContent = `Evidence-Based Source: ${data.technique.name} — ${data.technique.citation}`;
    document.getElementById("narrative").textContent = data.narrative;
    resultSection.classList.remove("hidden");

    // Audio narration player
    const audioPlayer = document.getElementById("audio-player");
    if (data.audio_url) {
      audioPlayer.src = data.audio_url;
      audioPlayer.classList.remove("hidden");
      audioPlayer.load();
    }

    // Comic strip polling
    pollImage(data.image_status_url);
  } catch (err) {
    loadingSection.classList.add("hidden");
    alert("Something went wrong: " + err.message);
  }
});

function pollImage(statusUrl) {
  const img = document.getElementById("result-image");
  const loadingText = document.getElementById("image-loading");
  img.classList.add("hidden");
  loadingText.classList.remove("hidden");

  const interval = setInterval(async () => {
    try {
      const res = await fetch(statusUrl);
      const data = await res.json();
      if (data.status === "done") {
        clearInterval(interval);
        img.src = data.image_url;
        img.classList.remove("hidden");
        loadingText.classList.add("hidden");
      } else if (data.status === "error") {
        clearInterval(interval);
        loadingText.textContent = "Comic generation failed: " + (data.error || "An error occurred during generation.");
      }
    } catch (e) {
      // Continue polling
    }
  }, 2000);
}
