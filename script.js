document.addEventListener('DOMContentLoaded', () => {

    const siteData = {
        totalReviews: 90,
        reviewsPerBrand: { Apple: 30, Samsung: 30, Google: 30 }
    };

    // ===================================================================
    // COLLEZ L'OBJET "phoneData" GÉNÉRÉ PAR VOTRE SCRIPT PYTHON ICI
    // Exemple de ce à quoi il devrait ressembler :
        const phoneData = {

    "Galaxy S22": {
        brand: "Samsung", score: 4.20,
        sentiments: { positive: 73, negative: 13, neutral: 13 },
        specs: {
            screen: "6.1\"", battery: "5000 mAh", camera: "48MP",
            price: "699€", ram: "6 GB RAM", storage: "128 GB",
            has_5g: "Non",
            water_resistant: "Non spécifié"
        },
        positiveCloud: "./images/wordcloud_Galaxy_S22_Positif_ai.png",
        negativeCloud: "./images/wordcloud_Galaxy_S22_Négatif_ai.png"
    },

    "Galaxy S23": {
        brand: "Samsung", score: 3.93,
        sentiments: { positive: 67, negative: 13, neutral: 20 },
        specs: {
            screen: "6.1\"", battery: "5000 mAh", camera: "108MP",
            price: "899€", ram: "12 GB RAM", storage: "64 GB",
            has_5g: "Non",
            water_resistant: "IP68"
        },
        positiveCloud: "./images/wordcloud_Galaxy_S23_Positif_ai.png",
        negativeCloud: "./images/wordcloud_Galaxy_S23_Négatif_ai.png"
    },

    "Pixel 7": {
        brand: "Google", score: 3.87,
        sentiments: { positive: 73, negative: 13, neutral: 13 },
        specs: {
            screen: "6.5\"", battery: "3000 mAh", camera: "108MP",
            price: "599€", ram: "6 GB RAM", storage: "128 GB",
            has_5g: "Non",
            water_resistant: "Non spécifié"
        },
        positiveCloud: "./images/wordcloud_Pixel_7_Positif_ai.png",
        negativeCloud: "./images/wordcloud_Pixel_7_Négatif_ai.png"
    },

    "Pixel 8": {
        brand: "Google", score: 4.13,
        sentiments: { positive: 67, negative: 13, neutral: 20 },
        specs: {
            screen: "6.5\"", battery: "3000 mAh", camera: "12MP",
            price: "799€", ram: "6 GB RAM", storage: "128 GB",
            has_5g: "Non",
            water_resistant: "IP68"
        },
        positiveCloud: "./images/wordcloud_Pixel_8_Positif_ai.png",
        negativeCloud: "./images/wordcloud_Pixel_8_Négatif_ai.png"
    },

    "iPhone 14": {
        brand: "Apple", score: 3.53,
        sentiments: { positive: 60, negative: 27, neutral: 13 },
        specs: {
            screen: "6.1\"", battery: "4000 mAh", camera: "48MP",
            price: "799€", ram: "6 GB RAM", storage: "128 GB",
            has_5g: "Oui",
            water_resistant: "Non spécifié"
        },
        positiveCloud: "./images/wordcloud_iPhone_14_Positif_ai.png",
        negativeCloud: "./images/wordcloud_iPhone_14_Négatif_ai.png"
    },

    "iPhone 15": {
        brand: "Apple", score: 3.80,
        sentiments: { positive: 67, negative: 20, neutral: 13 },
        specs: {
            screen: "6.1\"", battery: "3000 mAh", camera: "48MP",
            price: "999€", ram: "12 GB RAM", storage: "256 GB",
            has_5g: "Oui",
            water_resistant: "Non spécifié"
        },
        positiveCloud: "./images/wordcloud_iPhone_15_Positif_ai.png",
        negativeCloud: "./images/wordcloud_iPhone_15_Négatif_ai.png"
    },
};
    // ===================================================================

    const brandLogos = { Apple: "https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg", Samsung: "https://upload.wikimedia.org/wikipedia/commons/2/24/Samsung_Logo.svg", Google: "https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg" };

    const phoneSelector = document.getElementById('phone-selector');
    const phoneCard = document.getElementById('phone-card');

    function displayPhoneData(modelName) {
        const data = phoneData[modelName];
        if (!data) return;
        
        // La carte est maintenant beaucoup plus riche
        phoneCard.innerHTML = `
            <div class="card-header"><h2>${modelName}</h2><div class="brand-logo-container"><img src="${brandLogos[data.brand]}" alt="Logo ${data.brand}"></div></div>
            <div class="main-scores">
                <div class="score-card note-globale"><h3>Satisfaction Globale</h3><div class="score-value">${data.score.toFixed(2)}</div><div class="score-stars">${[...Array(5)].map((_, i) => `<span style="color: ${data.score > i ? '#ffc107' : '#e0e0e0'}">★</span>`).join('')}</div></div>
                <div class="score-card sentiment-dist"><h3>Distribution des Sentiments (IA)</h3><div class="sentiment-bars">
                    <div class="bar-label">Positifs</div><div class="bar-container"><div class="bar positive" style="width: ${data.sentiments.positive}%;"></div></div><div class="bar-value">${data.sentiments.positive}%</div>
                    <div class="bar-label">Neutres</div><div class="bar-container"><div class="bar neutral" style="width: ${data.sentiments.neutral}%;"></div></div><div class="bar-value">${data.sentiments.neutral}%</div>
                    <div class="bar-label">Négatifs</div><div class="bar-container"><div class="bar negative" style="width: ${data.sentiments.negative}%;"></div></div><div class="bar-value">${data.sentiments.negative}%</div>
                </div></div>
            </div>
            <div class="specs-section"><h3>Caractéristiques Techniques</h3><div class="specs-grid">
                <div class="spec-item"><span class="spec-icon">📱</span><div class="spec-details"><span class="spec-title">Écran</span><span class="spec-value">${data.specs.screen}</span></div></div>
                <div class="spec-item"><span class="spec-icon">🔋</span><div class="spec-details"><span class="spec-title">Batterie</span><span class="spec-value">${data.specs.battery}</span></div></div>
                <div class="spec-item"><span class="spec-icon">📸</span><div class="spec-details"><span class="spec-title">Appareil Photo</span><span class="spec-value">${data.specs.camera}</span></div></div>
                <div class="spec-item"><span class="spec-icon">💾</span><div class="spec-details"><span class="spec-title">Stockage</span><span class="spec-value">${data.specs.storage}</span></div></div>
                <div class="spec-item"><span class="spec-icon">⚡</span><div class="spec-details"><span class="spec-title">RAM</span><span class="spec-value">${data.specs.ram}</span></div></div>
                <div class="spec-item"><span class="spec-icon">📶</span><div class="spec-details"><span class="spec-title">5G</span><span class="spec-value">${data.specs.has_5g}</span></div></div>
                <div class="spec-item"><span class="spec-icon">💧</span><div class="spec-details"><span class="spec-title">Résistance Eau</span><span class="spec-value">${data.specs.water_resistant}</span></div></div>
                <div class="spec-item"><span class="spec-icon">💰</span><div class="spec-details"><span class="spec-title">Prix</span><span class="spec-value">${data.specs.price}</span></div></div>
            </div></div>
            <div class="analysis-grid">
                <div class="analysis-card"><h3>Ce que les gens aiment 👍</h3><img src="${data.positiveCloud}" alt="Nuage de mots positifs (IA)"></div>
                <div class="analysis-card"><h3>Points de vigilance 👎</h3><img src="${data.negativeCloud}" alt="Nuage de mots négatifs (IA)"></div>
            </div>
        `;
        phoneCard.classList.add('visible');

        document.querySelectorAll('#phone-selector button').forEach(btn => {
            btn.classList.toggle('active', btn.textContent === modelName);
        });
    }

    // Le reste de la fonction d'initialisation reste la même
    function initializeSite() {
        document.getElementById('total-reviews').textContent = siteData.totalReviews;
        document.getElementById('apple-reviews').textContent = siteData.reviewsPerBrand.Apple;
        document.getElementById('samsung-reviews').textContent = siteData.reviewsPerBrand.Samsung;
        document.getElementById('google-reviews').textContent = siteData.reviewsPerBrand.Google;

        const models = Object.keys(phoneData);
        models.forEach(modelName => {
            const button = document.createElement('button');
            button.textContent = modelName;
            button.addEventListener('click', () => displayPhoneData(modelName));
            phoneSelector.appendChild(button);
        });

        displayPhoneData(models[0]);
    }

    initializeSite();
});