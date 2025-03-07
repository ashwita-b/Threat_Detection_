async function fetchAlert() {
    const response = await fetch('/get_alert');
    const data = await response.json();
    document.getElementById('alert').innerText = data.alert;

    // Log the threat if detected
    if (data.alert !== "No threats detected") {
        const logBody = document.getElementById('log-body');
        const row = document.createElement('tr');
        row.innerHTML = `<td>${new Date().toLocaleTimeString()}</td><td>${data.alert}</td>`;

        // Insert at the top instead of appending to the bottom
        logBody.prepend(row);
    }
}

// Update alerts every second
setInterval(fetchAlert, 1000);

function startDetection() {
    document.getElementById("webcam").src = "/video_feed";
    alert("Threat detection started!");
}

function stopDetection() {
    document.getElementById("webcam").src = "";
    alert("Threat detection stopped!");
}

// Upload feature with real-time threat detection response
document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById('file-input');
    if (fileInput.files.length === 0) {
        alert("Please select a file before uploading.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch('/upload', {
            method: "POST",
            body: formData
        });

        const result = await response.json();
        alert(result.message);

        // Show processed image if available
        if (result.image_url) {
            const processedContainer = document.getElementById("processed-container");
            processedContainer.innerHTML = `<h3>Processed Image:</h3>
                                            <img src="${result.image_url}" alt="Processed Threat Image">`;
        }

        // Log detected threats in table
        if (result.message !== "No threats detected") {
            const logBody = document.getElementById('log-body');
            const row = document.createElement('tr');
            row.innerHTML = `<td>${new Date().toLocaleTimeString()}</td><td>${result.message}</td>`;
            logBody.prepend(row);
        }

    } catch (error) {
        console.error("Error uploading file:", error);
        alert("Error uploading file. Please try again.");
    }
});