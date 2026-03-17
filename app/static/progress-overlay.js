(function () {
    const STAGE_CARDS = [
        {
            key: "queued",
            emoji: "📥",
            badge: "Queued",
            title: "Job received",
            line: "Your video is queued and ready to start."
        },
        {
            key: "initializing",
            emoji: "🧭",
            badge: "Preparing",
            title: "Setting the pipeline",
            line: "Preparing model, tracker, and counting configuration."
        },
        {
            key: "counting",
            emoji: "🚗",
            badge: "Counting",
            title: "Tracking crossings",
            line: "Vehicles are being detected and counted now."
        },
        {
            key: "summarizing",
            emoji: "📊",
            badge: "Analytics",
            title: "Building insights",
            line: "Class totals and event summaries are being assembled."
        },
        {
            key: "rendering",
            emoji: "🎬",
            badge: "Rendering",
            title: "Rendering output",
            line: "Annotated preview video is being generated now."
        },
        {
            key: "completed",
            emoji: "✅",
            badge: "Done",
            title: "Analysis complete",
            line: "Everything is ready. Opening your results now."
        },
        {
            key: "failed",
            emoji: "⚠️",
            badge: "Failed",
            title: "Processing stopped",
            line: "The pipeline hit an error before finishing."
        }
    ];

    function getCardIndex(stageKey) {
        const foundIndex = STAGE_CARDS.findIndex((card) => card.key === stageKey);
        return foundIndex >= 0 ? foundIndex : 0;
    }

    function getStageCard(stageKey) {
        return STAGE_CARDS.find((card) => card.key === stageKey) || STAGE_CARDS[0];
    }

    function createOverlayMarkup() {
        return `
            <div class="processing-overlay" id="processingOverlay" hidden>
                <div class="processing-overlay-shell">
                    <div class="processing-status-chip" id="processingStatusChip">
                        <span class="processing-status-chip-dot"></span>
                        <span id="processingStatusChipText">Preparing live job status</span>
                    </div>

                    <div class="processing-carousel-frame">
                        <div class="processing-carousel-track" id="processingCarouselTrack">
                            ${STAGE_CARDS.map((card) => `
                                <article class="processing-card" data-stage-key="${card.key}">
                                    <div class="processing-card-top">
                                        <span class="processing-card-stage">${card.title}</span>
                                        <span class="processing-card-badge">${card.badge}</span>
                                    </div>

                                    <div class="processing-card-visual">
                                        <div class="processing-emoji-orb">
                                            <span class="processing-emoji">${card.emoji}</span>
                                        </div>
                                    </div>

                                    <div class="processing-card-text">${card.line}</div>
                                </article>
                            `).join("")}
                        </div>
                    </div>

                    <div class="processing-status-copy">
                        <h2 class="processing-status-title" id="processingStatusTitle">Processing your video</h2>
                        <p class="processing-status-subtitle" id="processingStatusSubtitle">
                            Initializing analysis workflow.
                        </p>
                    </div>

                    <div class="processing-progress-wrap">
                        <div class="processing-progress-meta">
                            <span id="processingStageLabel">Queued</span>
                            <span id="processingPercentLabel">0%</span>
                        </div>

                        <div class="processing-progress-shell" aria-hidden="true">
                            <div class="processing-progress-fill" id="processingProgressFill"></div>
                        </div>

                        <div class="processing-stage-label" id="processingMessageLabel">
                            Waiting to begin processing.
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    function initProcessingOverlay(config) {
        if (!config || !config.form || !config.fileInput) return;

        if (!document.getElementById("processingOverlay")) {
            document.body.insertAdjacentHTML("beforeend", createOverlayMarkup());
        }

        const overlay = document.getElementById("processingOverlay");
        const track = document.getElementById("processingCarouselTrack");
        const progressFill = document.getElementById("processingProgressFill");
        const percentLabel = document.getElementById("processingPercentLabel");
        const stageLabel = document.getElementById("processingStageLabel");
        const messageLabel = document.getElementById("processingMessageLabel");
        const statusTitle = document.getElementById("processingStatusTitle");
        const statusSubtitle = document.getElementById("processingStatusSubtitle");
        const statusChipText = document.getElementById("processingStatusChipText");

        let pollTimer = null;
        let lastKnownProgress = 0;
        let lastKnownStageKey = "queued";

        function stopPolling() {
            if (pollTimer) {
                window.clearInterval(pollTimer);
                pollTimer = null;
            }
        }

        function getVisibleCards() {
            if (window.innerWidth <= 640) return 1;
            if (window.innerWidth <= 900) return 2;
            return 3;
        }

        function setCardsActive(stageKey) {
            const cards = Array.from(track.querySelectorAll(".processing-card"));
            cards.forEach((card) => {
                card.classList.toggle("is-active", card.dataset.stageKey === stageKey);
            });

            const activeIndex = getCardIndex(stageKey);
            const activeCard = cards[activeIndex];
            if (!activeCard) return;

            const visibleCards = getVisibleCards();
            const cardWidth = activeCard.getBoundingClientRect().width + 16;

            let offsetIndex = Math.max(0, activeIndex - Math.floor((visibleCards - 1) / 2));
            const maxOffset = Math.max(0, cards.length - visibleCards);
            offsetIndex = Math.min(offsetIndex, maxOffset);

            track.style.transform = `translateX(-${offsetIndex * cardWidth}px)`;
        }

        function fadeOutAndRedirect(url) {
            overlay.classList.add("is-fading-out");
            window.setTimeout(() => {
                window.location.href = url;
            }, 520);
        }

        function showFailedState(message) {
            updateOverlay({
                stage_key: "failed",
                stage_label: "Failed",
                message: message || "Processing failed.",
                progress_pct: Math.max(lastKnownProgress, 100),
                status: "failed"
            });

            window.setTimeout(() => {
                overlay.hidden = true;
                overlay.classList.remove("is-fading-out");
                alert(message || "Processing failed.");
            }, 1200);
        }

        function updateOverlay(statusData) {
            const stageKey = statusData.stage_key || lastKnownStageKey || "queued";
            const stageName = statusData.stage_label || "Queued";
            const message = statusData.message || "Working on your request.";
            const progress = Math.max(lastKnownProgress, Number(statusData.progress_pct || 0));

            lastKnownProgress = progress;
            lastKnownStageKey = stageKey;

            const activeCard = getStageCard(stageKey);

            overlay.hidden = false;
            overlay.classList.remove("is-fading-out");

            progressFill.style.width = `${progress}%`;
            percentLabel.textContent = `${progress}%`;
            stageLabel.textContent = stageName;
            messageLabel.textContent = message;
            statusChipText.textContent = `${stageName} · live backend status`;

            statusTitle.textContent = activeCard.title;
            statusSubtitle.textContent = activeCard.line;

            setCardsActive(stageKey);
        }

        async function pollStatus(statusUrl, resultsUrl) {
            try {
                const response = await fetch(statusUrl, { cache: "no-store" });
                if (!response.ok) {
                    throw new Error(`Status polling failed: ${response.status}`);
                }

                const statusData = await response.json();
                updateOverlay(statusData);

                if (statusData.status === "completed") {
                    stopPolling();

                    updateOverlay({
                        ...statusData,
                        progress_pct: 100,
                        stage_key: "completed",
                        stage_label: "Done",
                        message: statusData.message || "Pipeline completed successfully."
                    });

                    window.setTimeout(() => {
                        fadeOutAndRedirect(statusData.results_url || resultsUrl);
                    }, 900);
                    return;
                }

                if (statusData.status === "failed") {
                    stopPolling();
                    showFailedState(statusData.message || "Processing failed.");
                }
            } catch (error) {
                stopPolling();
                overlay.hidden = true;
                overlay.classList.remove("is-fading-out");
                alert("Unable to read live processing progress. Please try again.");
                console.error(error);
            }
        }

        async function submitJob() {
            if (!config.fileInput.files || config.fileInput.files.length === 0) {
                alert("Please upload a video first.");
                return;
            }

            const formData = new FormData(config.form);

            lastKnownProgress = 0;
            lastKnownStageKey = "queued";

            updateOverlay({
                stage_key: "queued",
                stage_label: "Queued",
                progress_pct: 2,
                message: "Submitting job to backend.",
                status: "queued"
            });

            try {
                const response = await fetch("/api/jobs", {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`Job creation failed: ${response.status}`);
                }

                const payload = await response.json();

                if (!payload.status_url) {
                    throw new Error("Missing status_url from backend.");
                }

                await pollStatus(payload.status_url, payload.results_url);

                stopPolling();
                pollTimer = window.setInterval(() => {
                    pollStatus(payload.status_url, payload.results_url);
                }, 1200);
            } catch (error) {
                stopPolling();
                overlay.hidden = true;
                overlay.classList.remove("is-fading-out");
                alert("Unable to start processing. Please try again.");
                console.error(error);
            }
        }

        config.triggerButton?.addEventListener("click", (event) => {
            event.preventDefault();
            submitJob();
        });

        window.addEventListener("beforeunload", stopPolling);
        window.addEventListener("resize", () => {
            setCardsActive(lastKnownStageKey || "queued");
        });
    }

    window.initProcessingOverlay = initProcessingOverlay;
})();