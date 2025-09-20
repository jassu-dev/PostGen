document.addEventListener('DOMContentLoaded', function() {
    // Theme toggle
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;
    
    themeToggle.addEventListener('click', () => {
        body.classList.toggle('dark-theme');
        const icon = themeToggle.querySelector('i');
        if (body.classList.contains('dark-theme')) {
            icon.className = 'fas fa-sun';
        } else {
            icon.className = 'fas fa-moon';
        }
    });

    // Tab functionality
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;
            
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // Content generation
    const generateBtn = document.getElementById('generate-btn');
    const loadingIndicator = document.getElementById('loading-indicator');
    const outputSection = document.getElementById('output-section');
    const topicInput = document.getElementById('topic-input');
    const platformSelect = document.getElementById('platform-select');
    const toneSelect = document.getElementById('tone-select');
    const charLimitInput = document.getElementById('char-limit');
    const descriptionInput = document.getElementById('description');

    generateBtn.addEventListener('click', async () => {
        const topic = topicInput.value.trim();
        const platform = platformSelect.value;
        const tone = toneSelect.value;
        const charLimit = parseInt(charLimitInput.value);
        const description = descriptionInput.value.trim();

        if (!topic || !description) {
            alert('Please enter topic and description.');
            return;
        }

        if (charLimit < 50 || charLimit > 1000) {
            alert('Character limit must be between 50 and 1000.');
            return;
        }

        loadingIndicator.style.display = 'block';
        generateBtn.disabled = true;

        try {
            const response = await fetch('/api/generate_content', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    topic,
                    platform,
                    tone,
                    char_limit: charLimit,
                    description
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            displayGeneratedContent(data.content, platform);
        } catch (error) {
            console.error('Error generating content:', error);
            outputSection.innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>An error occurred while generating content. Please try again.</p>
                </div>
            `;
        } finally {
            loadingIndicator.style.display = 'none';
            generateBtn.disabled = false;
        }
    });

    function displayGeneratedContent(content, platform) {
        outputSection.innerHTML = `
            <div class="generated-content">
                <h3>Generated ${platform.charAt(0).toUpperCase() + platform.slice(1)} Post</h3>
                <div class="content-preview">
                    <p>${content}</p>
                </div>
                <div class="action-buttons">
                    <button class="copy-btn" onclick="copyToClipboard('${content}')">
                        <i class="fas fa-copy"></i> Copy to Clipboard
                    </button>
                    <button class="generate-images-btn" onclick="generateImages('${content}')">
                        <i class="fas fa-image"></i> Generate Images
                    </button>
                </div>
            </div>
        `;
    }

    window.copyToClipboard = function(text) {
        navigator.clipboard.writeText(text).then(() => {
            alert('Copied to clipboard!');
        }).catch(err => {
            console.error('Failed to copy: ', err);
            alert('Failed to copy to clipboard.');
        });
    };

    window.generateImages = async function(content) {
        const imageSection = document.createElement('div');
        imageSection.className = 'images-section';
        imageSection.innerHTML = `
            <h4>Generated Images for Your Post</h4>
            <div class="loading">
                <div class="spinner"></div>
                <p>Generating images...</p>
            </div>
            <div class="images-container" id="images-grid" style="display: none;"></div>
        `;
        outputSection.appendChild(imageSection);

        try {
            const response = await fetch('/api/generate_images', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: content, num_images: 4 }),
            });

            if (!response.ok) throw new Error('Failed to generate images');

            const data = await response.json();
            const grid = document.getElementById('images-grid');
            grid.innerHTML = '';
            data.images.forEach((imageUrl, index) => {
                const imgDiv = document.createElement('div');
                imgDiv.className = 'image-item';
                imgDiv.innerHTML = `
                    <img src="${imageUrl}" alt="Generated image ${index + 1}" onerror="this.src='/static/placeholder.jpg'">
                    <button class="download-btn" onclick="downloadImage('${imageUrl}', ${index + 1})">
                        <i class="fas fa-download"></i> Download
                    </button>
                `;
                grid.appendChild(imgDiv);
            });
            grid.style.display = 'grid';
            imageSection.querySelector('.loading').style.display = 'none';
        } catch (error) {
            console.error('Error generating images:', error);
            imageSection.innerHTML += '<p class="error-message">Failed to generate images. Image generation may not be configured.</p>';
        }
    };

    window.downloadImage = function(url, index) {
        const a = document.createElement('a');
        a.href = url;
        a.download = `post-image-${index}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };

    // AI Voiceovers
    const generateVoiceBtn = document.getElementById('generate-voice');
    const voicePreview = document.getElementById('voice-preview');
    const voiceStatus = document.getElementById('voice-status');
    const voiceText = document.getElementById('voice-text');
    const voiceStyle = document.getElementById('voice-style');

    generateVoiceBtn.addEventListener('click', async () => {
        const text = voiceText.value.trim();
        if (!text) {
            alert('Please enter text for voiceover.');
            return;
        }

        voiceStatus.innerHTML = '<div class="spinner"></div> Generating voice...';
        generateVoiceBtn.disabled = true;

        try {
            const response = await fetch('/api/generate_voice', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, voice: voiceStyle.value }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to generate voice');
            }

            const blob = await response.blob();
            const audioUrl = URL.createObjectURL(blob);
            voicePreview.src = audioUrl;
            voicePreview.style.display = 'block';
            voiceStatus.innerHTML = '<p class="success">Voice generated successfully!</p>';
        } catch (error) {
            console.error('Error generating voice:', error);
            voiceStatus.innerHTML = `<p class="error">Error: ${error.message}. TTS may not be configured.</p>`;
        } finally {
            generateVoiceBtn.disabled = false;
        }
    });

    // NFT Creator
    const mintNftBtn = document.getElementById('mint-nft');
    const nftContent = document.getElementById('nft-content');
    const nftStatus = document.getElementById('nft-status');
    const nftPreview = document.getElementById('nft-preview');
    const nftImageGrid = document.getElementById('nft-image-grid');

    mintNftBtn.addEventListener('click', async () => {
        const description = nftContent.value.trim();
        if (!description) {
            alert('Please enter a description for your NFT.');
            return;
        }

        nftStatus.innerHTML = '<div class="spinner"></div> Generating NFT image...';
        mintNftBtn.disabled = true;
        nftPreview.style.display = 'block';

        try {
            const response = await fetch('/api/generate_nft', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: `NFT style image: ${description}` }),
            });

            if (!response.ok) throw new Error('Failed to generate NFT image');

            const data = await response.json();
            nftImageGrid.innerHTML = `
                <div class="image-item">
                    <img src="${data.image_url}" alt="NFT Image" onerror="this.src='/static/placeholder.jpg'">
                    <button class="download-btn" onclick="downloadImage('${data.image_url}', 'nft')">
                        <i class="fas fa-download"></i> Download NFT
                    </button>
                </div>
            `;
            nftStatus.innerHTML = '<p class="success">NFT image generated! (Demo - not minted on blockchain)</p>';
        } catch (error) {
            console.error('Error generating NFT:', error);
            nftStatus.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        } finally {
            mintNftBtn.disabled = false;
        }
    });

    // Analytics
    const analyzeBtn = document.getElementById('analyze-btn');
    const analyticsLoading = document.getElementById('analytics-loading');
    const analyticsResults = document.getElementById('analytics-results');
    const postContent = document.getElementById('post-content');
    const analyticsPlatform = document.getElementById('analytics-platform');
    let engagementChart = null;

    analyzeBtn.addEventListener('click', async () => {
        const content = postContent.value.trim();
        const platform = analyticsPlatform.value;

        if (!content) {
            alert('Please paste your post content.');
            return;
        }

        analyticsLoading.style.display = 'block';
        analyticsResults.style.display = 'none';
        analyzeBtn.disabled = true;

        try {
            const response = await fetch('/api/analyze_post', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ post_text: content, platform }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            displayAnalyticsResults(data, platform);
        } catch (error) {
            console.error('Error analyzing post:', error);
            analyticsResults.innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>An error occurred while analyzing. Please try again.</p>
                </div>
            `;
            analyticsResults.style.display = 'block';
        } finally {
            analyticsLoading.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });

    function displayAnalyticsResults(data, platform) {
        // Score
        document.getElementById('post-score').innerHTML = `
            <span class="score-number">${data.score}/100</span>
            <div class="score-bar">
                <div class="score-fill" style="width: ${data.score}%"></div>
            </div>
        `;
        document.getElementById('score-explanation').textContent = data.score_explanation;

        // Hide or clear engagement section - no estimations
        const engagementSection = document.querySelector('.engagement-section');
        if (engagementSection) {
            engagementSection.style.display = 'none';
        }

        // Post analysis
        document.getElementById('post-analysis').innerHTML = `<p>${data.analysis}</p>`;

        // Improvements (main focus)
        const improvementsList = document.getElementById('improvements-list');
        improvementsList.innerHTML = '';
        if (data.improvements && Array.isArray(data.improvements)) {
            data.improvements.forEach(imp => {
                const li = document.createElement('li');
                li.textContent = imp;
                improvementsList.appendChild(li);
            });
        } else {
            improvementsList.innerHTML = '<li>No specific improvements available.</li>';
        }

        // Better post
        document.getElementById('better-post').innerHTML = `<p>${data.better_post || 'No improved version available.'}</p>`;

        // Competitors (main focus)
        const competitorsList = document.getElementById('competitors-list');
        competitorsList.innerHTML = '<h5>Learn from these effective competitor posts:</h5>';
        if (data.competitor_examples && Array.isArray(data.competitor_examples)) {
            data.competitor_examples.forEach((example, index) => {
                const why = data.competitor_why && data.competitor_why[index] ? data.competitor_why[index] : 'High engagement due to concise, engaging format.';
                const div = document.createElement('div');
                div.className = 'competitor-example';
                div.innerHTML = `
                    <h6>Example ${index + 1}:</h6>
                    <p class="content-preview">${example}</p>
                    <p><em>Why it works: ${why}</em></p>
                `;
                competitorsList.appendChild(div);
            });
        } else {
            competitorsList.innerHTML += '<p>No competitor examples available at this time.</p>';
        }

        // Update platform in header
        document.querySelector('.competitors-section h4').innerHTML = `Effective Competitor Examples on ${platform.charAt(0).toUpperCase() + platform.slice(1)}`;

        analyticsResults.style.display = 'block';
    }

    // Meme Generator (if needed, but not in current HTML - can add later)
    // Placeholder for meme functionality if expanded
});