/**
 * Aircraft Lever Operator Monitoring System
 * Universal Device Discovery & Telemetry Controller (Chart.js + REST API)
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- UI Elements ---
    const btnStart = document.getElementById("btnStart");
    const btnStop = document.getElementById("btnStop");
    const timerVal = document.getElementById("timerVal");

    // Header Status Indicators
    const dotCamera = document.getElementById("dotCamera");
    const dotForce = document.getElementById("dotForce");

    // Live Video & Camera Stats
    const videoFeed = document.getElementById("videoFeed");
    const videoPlaceholder = document.getElementById("videoPlaceholder");
    const lblActiveCamera = document.getElementById("lblActiveCamera");
    const trackingStatus = document.getElementById("trackingStatus");
    const resDisplay = document.getElementById("resDisplay");
    const fpsDisplay = document.getElementById("fpsDisplay");

    // Camera Calculation Metric Values
    const valWristAngle = document.getElementById("valWristAngle");
    const valWristDev = document.getElementById("valWristDev");
    const valMovAngle = document.getElementById("valMovAngle");
    const valSpeed = document.getElementById("valSpeed");
    const valRange = document.getElementById("valRange");

    // Hardware Force Metric Values
    const valForceCurrent = document.getElementById("valForceCurrent");
    const valForceAvg = document.getElementById("valForceAvg");
    const valForcePeak = document.getElementById("valForcePeak");
    const lblForceType = document.getElementById("lblForceType");
    const lblForceDevice = document.getElementById("lblForceDevice");
    const lblForceStatus = document.getElementById("lblForceStatus");

    // Universal Device Configuration Controls
    const btnRefreshDevices = document.getElementById("btnRefreshDevices");
    const cfgCamName = document.getElementById("cfgCamName");
    const cfgCamInterface = document.getElementById("cfgCamInterface");
    const cfgCamDevice = document.getElementById("cfgCamDevice");
    const cfgCamResolution = document.getElementById("cfgCamResolution");
    const cfgCamStatus = document.getElementById("cfgCamStatus");
    const tblExternalDevicesBody = document.getElementById("tblExternalDevicesBody");

    const btnModes = document.querySelectorAll(".btn-mode");
    const lblNetworkDiscovery = document.getElementById("lblNetworkDiscovery");
    const rowSerialSelect = document.getElementById("rowSerialSelect");
    const selSerialPorts = document.getElementById("selSerialPorts");
    const btnConnectSerial = document.getElementById("btnConnectSerial");
    const btnDisconnectSerial = document.getElementById("btnDisconnectSerial");

    const rowWifiConfig = document.getElementById("rowWifiConfig");
    const txtEspIp = document.getElementById("txtEspIp");
    const btnConnectWifi = document.getElementById("btnConnectWifi");

    // --- State & Settings ---
    let isRunning = false;
    let pollInterval = null;
    const MAX_CHART_POINTS = 120; // ~30-40 seconds rolling window at 3-4 Hz

    // --- Chart.js Configuration & Theme ---
    Chart.defaults.color = "#4b5563";
    Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    Chart.defaults.font.size = 11;
    Chart.defaults.plugins.legend.display = false;

    const makeChartScales = (xTitle, yTitle, yMin = null, yMax = null) => ({
        x: {
            title: { display: true, text: xTitle, color: "#4b5563", font: { size: 10, weight: "bold" } },
            grid: { color: "rgba(0, 0, 0, 0.05)" },
            ticks: { maxTicksLimit: 6 }
        },
        y: {
            title: { display: true, text: yTitle, color: "#4b5563", font: { size: 10, weight: "bold" } },
            grid: { color: "rgba(0, 0, 0, 0.06)" },
            min: yMin,
            max: yMax,
            ticks: { maxTicksLimit: 5 }
        }
    });

    // 1. Grip Force vs Time
    const chartGripForce = new Chart(document.getElementById("chartGripForce"), {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "Grip Force (%)",
                data: [],
                borderColor: "#2563eb",
                backgroundColor: "rgba(37, 99, 235, 0.1)",
                fill: true,
                tension: 0.15,
                pointRadius: 0,
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: makeChartScales("Elapsed Time (s)", "Grip Force (%)", 0, 100)
        }
    });

    // 2. Wrist Angle vs Time
    const chartWristAngle = new Chart(document.getElementById("chartWristAngle"), {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "Wrist Angle (°)",
                data: [],
                borderColor: "#f59e0b",
                backgroundColor: "rgba(245, 158, 11, 0.1)",
                fill: true,
                tension: 0.15,
                pointRadius: 0,
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: makeChartScales("Elapsed Time (s)", "Wrist Angle (°)", null, null)
        }
    });

    // 3. Movement Angle vs Time
    const chartMovementAngle = new Chart(document.getElementById("chartMovementAngle"), {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "Movement Angle (°)",
                data: [],
                borderColor: "#10b981",
                backgroundColor: "rgba(16, 185, 129, 0.1)",
                fill: true,
                tension: 0.15,
                pointRadius: 0,
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: makeChartScales("Elapsed Time (s)", "Movement Angle (°)", null, null)
        }
    });

    // 4. Angular Speed vs Time
    const chartAngularSpeed = new Chart(document.getElementById("chartAngularSpeed"), {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "Angular Speed (°/s)",
                data: [],
                borderColor: "#06b6d4",
                backgroundColor: "rgba(6, 182, 212, 0.1)",
                fill: true,
                tension: 0.15,
                pointRadius: 0,
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: makeChartScales("Elapsed Time (s)", "Angular Speed (°/s)", 0, null)
        }
    });

    function resetCharts() {
        [chartGripForce, chartWristAngle, chartMovementAngle, chartAngularSpeed].forEach(c => {
            c.data.labels = [];
            c.data.datasets[0].data = [];
            c.update();
        });
    }

    function appendChartData(chart, timeLabel, value) {
        chart.data.labels.push(timeLabel);
        chart.data.datasets[0].data.push(value);
        if (chart.data.labels.length > MAX_CHART_POINTS) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        chart.update();
    }

    // --- Universal Device Discovery (Connected Camera, External Devices, Serial COM, Network) ---
    async function scanAllDevices() {
        try {
            if (btnRefreshDevices) {
                btnRefreshDevices.disabled = true;
                btnRefreshDevices.textContent = "Refreshing...";
            }
            const resp = await fetch("/api/devices");
            const data = await resp.json();

            // 1. Populate Connected Camera Card
            const cam = data.camera || {};
            const camName = cam.device_name || data.selected_camera?.name || "Logitech HD Webcam C270";
            if (cfgCamName) cfgCamName.textContent = camName;
            if (cfgCamInterface) cfgCamInterface.textContent = cam.interface || "USB Camera";
            if (cfgCamDevice) cfgCamDevice.textContent = cam.analysis_device || "Camera 0";
            if (cfgCamResolution) cfgCamResolution.textContent = cam.resolution || "1280 × 720";
            if (cfgCamStatus) {
                cfgCamStatus.textContent = isRunning ? "ACTIVE" : (cam.status || "CONNECTED");
                cfgCamStatus.className = "info-val text-success";
            }
            if (lblActiveCamera) lblActiveCamera.textContent = camName;

            // 2. Populate Connected External Devices Table
            if (tblExternalDevicesBody) {
                tblExternalDevicesBody.innerHTML = "";
                const extDevs = data.external_devices || [];
                if (extDevs.length === 0) {
                    tblExternalDevicesBody.innerHTML = `<tr><td colspan="4" class="text-center text-muted">No external devices detected</td></tr>`;
                } else {
                    extDevs.forEach(d => {
                        const tr = document.createElement("tr");
                        let typeClass = "other";
                        const tLower = (d.type || "").toLowerCase();
                        if (tLower.includes("camera")) typeClass = "camera";
                        else if (tLower.includes("input")) typeClass = "input";
                        else if (tLower.includes("serial")) typeClass = "serial";
                        
                        tr.innerHTML = `
                            <td><strong>${d.name}</strong></td>
                            <td><span class="dev-type-badge ${typeClass}">${d.type}</span></td>
                            <td>${d.connection || "USB"}</td>
                            <td><span class="dev-status-badge">${d.status || "CONNECTED"}</span></td>
                        `;
                        tblExternalDevicesBody.appendChild(tr);
                    });
                }
            }

            // 3. Populate Serial Devices
            if (selSerialPorts) {
                selSerialPorts.innerHTML = "";
                const serials = data.serial_devices || [];
                if (serials.length === 0) {
                    selSerialPorts.innerHTML = '<option value="">No serial devices detected</option>';
                } else {
                    serials.forEach(s => {
                        const opt = document.createElement("option");
                        opt.value = s.port;
                        const meta = s.vid_pid ? ` [${s.vid_pid}]` : "";
                        opt.textContent = `${s.port} — ${s.description}${meta}`;
                        selSerialPorts.appendChild(opt);
                    });
                }
            }

            // 4. Populate Network Device
            if (lblNetworkDiscovery) {
                const net = data.network_device || {};
                const netStatus = net.status || "Not Found";
                const isConn = net.connected === true;
                lblNetworkDiscovery.innerHTML = `${net.name || "ESP32"} (${net.address || "192.168.4.1"}) — <strong style="color:${isConn ? '#059669' : '#6b7280'}">${netStatus}</strong>`;
            }

        } catch (err) {
            console.error("Device scan error:", err);
        } finally {
            if (btnRefreshDevices) {
                btnRefreshDevices.disabled = isRunning;
                btnRefreshDevices.textContent = "REFRESH DEVICES";
            }
        }
    }

    if (btnRefreshDevices) {
        btnRefreshDevices.addEventListener("click", scanAllDevices);
    }

    // --- Unified Live Polling Loop (~3.3 Hz) ---
    async function pollLiveTelemetry() {
        try {
            const res = await fetch("/api/live");
            if (!res.ok) return;
            const data = await res.json();

            // 1. Header Indicators
            const camConnected = data.camera.state === "CONNECTED";
            dotCamera.className = "indicator-dot" + (camConnected ? " active" : (isRunning ? " error" : ""));

            const forceConnected = data.force.state === "CONNECTED";
            dotForce.className = "indicator-dot" + (forceConnected ? " active" : "");

            // 2. Timer
            timerVal.textContent = data.elapsed_formatted || "00:00";

            // 3. Live Video Stats
            lblActiveCamera.textContent = data.camera.device_name || `Camera ${data.camera.device_index || 0}`;
            trackingStatus.textContent = data.camera.tracking || "WAITING";
            trackingStatus.style.color = data.camera.tracking === "GOOD" ? "#059669" : (data.camera.tracking === "LIMITED" ? "#d97706" : "#4b5563");
            resDisplay.textContent = data.camera.resolution || "1280x720";
            fpsDisplay.textContent = (data.camera.fps || 0).toFixed(1);

            // 4. Camera Calculations
            const ang = data.angles;
            valWristAngle.innerHTML = ang.wrist_angle !== null && ang.wrist_angle !== undefined 
                ? `${ang.wrist_angle > 0 ? "+" : ""}${ang.wrist_angle.toFixed(1)} <span class="metric-unit">°</span>` 
                : `-- <span class="metric-unit">°</span>`;

            valWristDev.innerHTML = ang.wrist_deviation !== null && ang.wrist_deviation !== undefined 
                ? `${ang.wrist_deviation.toFixed(1)} <span class="metric-unit">°</span>` 
                : `-- <span class="metric-unit">°</span>`;

            valMovAngle.innerHTML = ang.movement_angle !== null && ang.movement_angle !== undefined 
                ? `${ang.movement_angle > 0 ? "+" : ""}${ang.movement_angle.toFixed(1)} <span class="metric-unit">°</span>` 
                : `-- <span class="metric-unit">°</span>`;

            valSpeed.innerHTML = ang.angular_speed !== null && ang.angular_speed !== undefined 
                ? `${ang.angular_speed.toFixed(1)} <span class="metric-unit">°/s</span>` 
                : `-- <span class="metric-unit">°/s</span>`;

            valRange.innerHTML = ang.movement_range !== null && ang.movement_range !== undefined 
                ? `${ang.movement_range.toFixed(1)} <span class="metric-unit">°</span>` 
                : `-- <span class="metric-unit">°</span>`;

            // 5. Hardware Force
            const f = data.force;
            valForceCurrent.innerHTML = f.current !== null && f.current !== undefined 
                ? `${f.current.toFixed(1)} <span class="metric-unit">%</span>` 
                : `-- <span class="metric-unit">%</span>`;

            valForceAvg.innerHTML = f.average !== null && f.average !== undefined 
                ? `${f.average.toFixed(1)} <span class="metric-unit">%</span>` 
                : `-- <span class="metric-unit">%</span>`;

            valForcePeak.innerHTML = f.peak !== null && f.peak !== undefined 
                ? `${f.peak.toFixed(1)} <span class="metric-unit">%</span>` 
                : `-- <span class="metric-unit">%</span>`;

            lblForceType.textContent = f.connection_type || "None";
            lblForceDevice.textContent = f.device || "None";
            const forceConn = f.state === "CONNECTED";
            lblForceStatus.textContent = forceConn ? "CONNECTED" : "Not Connected";
            lblForceStatus.style.color = forceConn ? "#059669" : "#6b7280";

            // 6. Push Live Graph Data
            if (isRunning && data.elapsed_time !== undefined) {
                const tStr = data.elapsed_time.toFixed(1);
                appendChartData(chartGripForce, tStr, f.current);
                appendChartData(chartWristAngle, tStr, ang.wrist_angle);
                appendChartData(chartMovementAngle, tStr, ang.movement_angle);
                appendChartData(chartAngularSpeed, tStr, ang.angular_speed);
            }

        } catch (e) {
            console.error("Telemetry error:", e);
        }
    }

    // --- Master Controls ---
    btnStart.addEventListener("click", async () => {
        btnStart.disabled = true;
        try {
            const resp = await fetch("/api/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({})
            });
            const res = await resp.json();
            if (res.status === "ok") {
                isRunning = true;
                btnStop.disabled = false;
                if (btnRefreshDevices) btnRefreshDevices.disabled = true;
                if (cfgCamStatus) {
                    cfgCamStatus.textContent = "ACTIVE";
                }

                // Connect video stream with anti-cache query
                videoPlaceholder.style.display = "none";
                videoFeed.style.display = "block";
                videoFeed.src = `/video_feed?t=${Date.now()}`;

                resetCharts();
            } else {
                btnStart.disabled = false;
            }
        } catch (err) {
            console.error("Start failed:", err);
            btnStart.disabled = false;
        }
    });

    btnStop.addEventListener("click", async () => {
        btnStop.disabled = true;
        try {
            const resp = await fetch("/api/stop", { method: "POST" });
            const res = await resp.json();
            if (res.status === "ok") {
                isRunning = false;
                btnStart.disabled = false;
                if (btnRefreshDevices) btnRefreshDevices.disabled = false;
                if (cfgCamStatus) {
                    cfgCamStatus.textContent = "CONNECTED";
                }

                // Stop video feed
                videoFeed.src = "";
                videoFeed.style.display = "none";
                videoPlaceholder.style.display = "flex";
            }
        } catch (err) {
            console.error("Stop failed:", err);
            btnStart.disabled = false;
        }
    });

    // --- Force Source Mode Switching ---
    btnModes.forEach(btn => {
        btn.addEventListener("click", async () => {
            btnModes.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const mode = btn.dataset.mode;

            if (mode === "WIFI") {
                rowWifiConfig.style.display = "flex";
                rowSerialSelect.style.display = "none";
            } else if (mode === "SERIAL") {
                rowWifiConfig.style.display = "none";
                rowSerialSelect.style.display = "flex";
            } else { // AUTO
                rowWifiConfig.style.display = "none";
                rowSerialSelect.style.display = "flex";
            }

            await fetch("/api/config_sensor", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode: mode })
            });
        });
    });

    btnConnectSerial.addEventListener("click", async () => {
        const port = selSerialPorts.value;
        if (!port) return;
        await fetch("/api/config_sensor", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: "SERIAL", port: port, action: "connect" })
        });
    });

    btnDisconnectSerial.addEventListener("click", async () => {
        await fetch("/api/config_sensor", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "disconnect" })
        });
    });

    btnConnectWifi.addEventListener("click", async () => {
        const ip = txtEspIp.value.trim();
        if (!ip) return;
        await fetch("/api/config_sensor", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: "WIFI", ip: ip, action: "connect" })
        });
    });

    // Initial Universal Device Discovery & Telemetry Heartbeat
    scanAllDevices();
    pollInterval = setInterval(pollLiveTelemetry, 300);
    pollLiveTelemetry();
});
