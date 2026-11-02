(function () {
    const form = document.getElementById("wineForm");
    const sliders = Array.from(form.querySelectorAll('input[type="range"]'));
    const toggles = Array.from(document.querySelectorAll(".toggle-btn"));
    const predictBtn = document.getElementById("predictBtn");
    let wineType = "red";

    function decimals(slider) {
        const range = parseFloat(slider.max) - parseFloat(slider.min);
        if (range < 0.05) return 4;
        if (range < 1) return 3;
        if (range < 20) return 2;
        return 0;
    }

    function showValue(slider) {
        slider.nextElementSibling.textContent = parseFloat(slider.value).toFixed(decimals(slider));
    }

    function resetToTypical() {
        sliders.forEach((s) => {
            s.value = s.dataset[wineType];
            showValue(s);
        });
    }

    sliders.forEach((s) => {
        s.addEventListener("input", () => showValue(s));
        showValue(s);
    });

    toggles.forEach((btn) => {
        btn.addEventListener("click", () => {
            wineType = btn.dataset.type;
            toggles.forEach((b) => b.classList.toggle("active", b === btn));
            resetToTypical();
        });
    });

    document.getElementById("resetBtn").addEventListener("click", resetToTypical);

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const payload = { wine_type: wineType };
        sliders.forEach((s) => { payload[s.name] = parseFloat(s.value); });

        predictBtn.disabled = true;
        predictBtn.textContent = "Tasting...";
        const errorBox = document.getElementById("resultError");
        errorBox.hidden = true;

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Prediction failed");
            render(data);
        } catch (err) {
            errorBox.textContent = err.message;
            errorBox.hidden = false;
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = "Rate this wine";
        }
    });

    function render(data) {
        document.getElementById("resultEmpty").hidden = true;
        document.getElementById("resultBody").hidden = false;

        const verdict = document.getElementById("verdict");
        verdict.textContent = data.predicted_class;
        verdict.className = "badge cls-" + data.predicted_class.toLowerCase();
        document.getElementById("confidence").textContent =
            Math.round(data.confidence * 100) + "% confident";

        Object.entries(data.probabilities).forEach(([cls, p]) => {
            document.querySelector(`.bar-fill[data-class="${cls}"]`).style.width = (p * 100).toFixed(1) + "%";
            document.querySelector(`.bar-value[data-value="${cls}"]`).textContent = Math.round(p * 100) + "%";
        });

        document.getElementById("logNote").textContent =
            `Saved as prediction #${data.id} (${data.wine_type} wine). See it on the History page.`;
    }
})();
