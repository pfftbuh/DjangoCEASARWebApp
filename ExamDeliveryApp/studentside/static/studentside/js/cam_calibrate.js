document.addEventListener('DOMContentLoaded', function() {
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;
    const gazeCircle = document.getElementById('gaze-circle');
    const calibrationPoint = document.getElementById('calibration-point');
    const startExamBtn = document.getElementById('start-exam-btn');
    const loadCalibrationBtn = document.getElementById('load-calibration-btn');
    const resetCalibrationBtn = document.getElementById('reset-calibration-btn');
    const cameraFeed = document.getElementById('camera-feed');
    const container = document.querySelector('.container-fluid');

    let calibrationStatusInterval;
    let currentStage;

    // Get CSRF token
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    const csrftoken = getCookie('csrftoken');

    // Get URLs from data attributes
    const urls = {
        updateCalibration: container.dataset.updateCalibrationUrl,
        saveCalibration: container.dataset.saveCalibrationUrl,
        getScreenPosition: container.dataset.getScreenPositionUrl,
        getFaceDistance: container.dataset.getFaceDistanceUrl,
        getCalibrationStage: container.dataset.getCalibrationStageUrl,
        updateCalibrationStatus: container.dataset.updateCalibrationStatusUrl,
        releaseCamera: container.dataset.releaseCameraUrl,
        resetCalibration: container.dataset.resetCalibrationUrl,
        loadCalibrationData: container.dataset.loadCalibrationDataUrl,
        startExamUrl: container.dataset.startExamUrl
    };

    // Calibration point positions
    const calibrationPoints = {
        '-1': { x: screenWidth / 2, y: screenHeight * 0.05, text: "Look at: Center Up" },
        '0': { x: screenWidth / 2, y: screenHeight * 0.45, text: "Look at: Center" },
        '1': { x: screenWidth / 2, y: screenHeight * 0.85, text: "Look at: Center Down" },
        '2': { x: screenWidth * 0.10, y: screenHeight / 2, text: "Look at: Left Center" },
        '3': { x: screenWidth / 2, y: screenHeight * 0.45, text: "Look at: Center" },
        '4': { x: screenWidth * 0.90, y: screenHeight / 2, text: "Look at: Right Center" },
        '5': { x: screenWidth / 2, y: screenHeight / 2, text: "Calibration Complete!" },
        '6': { x: 0, y: 0, text: "Tracking Active" }
    };

    function showCalibrationPoint(stage) {
        const point = calibrationPoints[stage];
        if (point && stage < 6) {
            calibrationPoint.style.left = point.x + 'px';
            calibrationPoint.style.top = point.y + 'px';
            calibrationPoint.style.display = 'block';
        } else {
            calibrationPoint.style.display = 'none';
        }
    }

    function updateProgressBar(stage) {
        const progress = Math.min(100, ((stage + 2) / 7) * 100);
        const progressBar = document.getElementById('calibration-progress');
        progressBar.style.width = progress + '%';
        progressBar.setAttribute('aria-valuenow', progress);
        progressBar.textContent = Math.round(progress) + '%';
        
        if (stage >= 6) {
            progressBar.classList.remove('progress-bar-animated');
            progressBar.classList.add('bg-success');
            startExamBtn.disabled = false;
        }
    }

    document.addEventListener('keydown', function(event) {
        if (event.key === 'c' || event.key === 'C') {
            // Check if the border is green (calibrated) before allowing progression
            const isGreen = cameraFeed.style.borderColor === 'green';
            
            if (!isGreen && currentStage < 6) {
                // Show visual feedback that calibration is not complete
                const statusBadge = document.querySelector('#calibration-status .badge');
                const originalText = statusBadge.textContent;
                
                statusBadge.textContent = "⚠️ Wait for GREEN border!";
                statusBadge.classList.remove('bg-secondary', 'bg-success');
                statusBadge.classList.add('bg-warning');
                
                // Flash the border orange
                cameraFeed.style.borderColor = 'orange';
                setTimeout(() => {
                    cameraFeed.style.borderColor = 'red';
                    statusBadge.textContent = originalText;
                    statusBadge.classList.remove('bg-warning');
                    statusBadge.classList.add('bg-secondary');
                }, 1000);
                
                return; // Don't proceed to next stage
            }
            
            // Border is green, proceed to next calibration stage
            fetch(urls.updateCalibration, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrftoken,
                },
            })
            .then(response => response.json())
            .then(data => {
                console.log('Calibration updated:', data);
                currentStage = data.calibration_stage;
                
                const point = calibrationPoints[currentStage];
                document.getElementById('calibration-stage').textContent = 
                    `Calibration Stage: ${data.calibration_stage} - ${point ? point.text : 'Unknown'}`;
                
                updateProgressBar(currentStage);
                showCalibrationPoint(currentStage);  // Move to next point
                
                // Stop calibration status updates when stage 6 is reached
                if (data.calibration_stage >= 6 && calibrationStatusInterval) {
                    clearInterval(calibrationStatusInterval);
                    save_calibration();
                    console.log('Calibration complete - stopped status updates');
                }
            })
            .catch(error => console.error('Error updating calibration:', error));
        }
    });


    function save_calibration() {
        fetch(urls.saveCalibration, {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrftoken,
            },
        })
        .then(response => response.json())
        .then(data => {
            console.log('Calibration saved:', data);
        })
        .catch(error => console.error('Error saving calibration:', error));
    }

    function fetchGazePoint() {
        fetch(urls.getScreenPosition)
            .then(response => response.json())
            .then(data => {
                document.getElementById('mid-x').textContent = data.mid_x.toFixed(2);
                document.getElementById('mid-y').textContent = data.mid_y.toFixed(2);
                
                // Data is already normalized (0-1), scale to screen dimensions
                const scaledX = data.mid_x * screenWidth;
                const scaledY = data.mid_y * screenHeight;
                
                // Clamp to screen bounds
                const clampedX = Math.max(0, Math.min(scaledX, screenWidth));
                const clampedY = Math.max(0, Math.min(scaledY, screenHeight));
                
                // Update circle position
                gazeCircle.style.left = clampedX + 'px';
                gazeCircle.style.top = clampedY + 'px';
                gazeCircle.style.display = 'block';
            })
            .catch(error => console.error('Error fetching eye midpoint:', error));
    }

    // Fetch face distance every 200ms
    function fetchFaceDistance(){
        fetch(urls.getFaceDistance)
            .then(response => response.json())
            .then(data => {
                console.log('Face distance (cm):', data.face_distance_cm);
                document.getElementById('face-distance').textContent = data.face_distance_cm.toFixed(2) + ' cm';
                const distanceInfo = document.getElementById('distance-info');
                // Update distance info class based on distance
                if (data.face_distance_cm >= 40 && data.face_distance_cm <= 60) {
                    distanceInfo.classList.remove('alert-warning');
                    distanceInfo.classList.add('alert-info');
                } else {
                    distanceInfo.classList.remove('alert-info');
                    distanceInfo.classList.add('alert-warning');
                }
            })
            .catch(error => console.error('Error fetching face distance:', error));
    }

    setInterval(fetchFaceDistance, 200);

    // Fetch initial calibration stage FIRST, then start everything else
    fetch(urls.getCalibrationStage)
        .then(response => response.json())
        .then(data => {
            currentStage = data.calibration_stage;
            const point = calibrationPoints[currentStage];
            document.getElementById('calibration-stage').textContent = 
                `Calibration Stage: ${data.calibration_stage} - ${point ? point.text : 'Unknown'}`;
            updateProgressBar(currentStage);
            showCalibrationPoint(currentStage);
            
            // NOW start the intervals after we have currentStage
            // Fetch eye midpoint every 50ms
            setInterval(fetchGazePoint, 50);
            
            // Update calibration status every 200ms
            calibrationStatusInterval = setInterval(updateCalibrationStatus, 200);
        })
        .catch(error => {
            console.error('Error fetching initial calibration stage:', error);
            // Fallback to -1 if error
            currentStage = -1;
            showCalibrationPoint(currentStage);
            
            // Start intervals anyway
            setInterval(fetchGazePoint, 200);
            calibrationStatusInterval = setInterval(updateCalibrationStatus, 200);
        });

    function updateCalibrationStatus() {
        fetch(urls.updateCalibrationStatus, {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrftoken,
            },
        }).then(response => response.json())
        .then(data => {
            const statusBadge = document.querySelector('#calibration-status .badge');
            
            if (data.status === true || data.status === 'true') {
                statusBadge.textContent = "Status: Calibrated";
                statusBadge.classList.remove('bg-secondary');
                statusBadge.classList.add('bg-success');
                cameraFeed.style.borderColor = 'green';
            } else {
                statusBadge.textContent = "Status: Not Calibrated";
                statusBadge.classList.remove('bg-success');
                statusBadge.classList.add('bg-secondary');
                cameraFeed.style.borderColor = 'red';
            }
        })
        .catch(error => console.error('Error updating calibration status:', error));
    }
    

    // Handle page unload - use beforeunload for better reliability
    window.addEventListener('beforeunload', function(e) {
        // Clear interval first
        clearInterval(calibrationStatusInterval);
        
        // Release camera with keepalive to ensure request completes
        fetch(urls.releaseCamera, {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrftoken,
            },
            keepalive: true  // Ensures request completes even if page closes
        });
        
        // Reset calibration
        fetch(urls.resetCalibration, {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrftoken,
            },
            keepalive: true
        });
    });

    // Also handle visibility change (when tab is hidden/closed)
    document.addEventListener('visibilitychange', function() {
        if (document.visibilityState === 'hidden') {
            fetch(urls.releaseCamera, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrftoken,
                },
                keepalive: true
            }).catch(error => console.error('Error releasing camera:', error));
        }
    });

    // Load existing calibration data
    loadCalibrationBtn.addEventListener('click', function() {
        fetch(urls.loadCalibrationData, {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrftoken,
            },
        })
        .then(response => response.json())
        .then(data => {
            console.log('Calibration loaded:', data);
            showCalibrationPoint(data.currentStage);
            updateProgressBar(data.currentStage);
            currentStage = data.currentStage;
            alert('Existing calibration data loaded successfully.');
        })
        .catch(error => console.error('Error loading calibration:', error));
    });

    // Reset calibration data
    resetCalibrationBtn.addEventListener('click', function() {
        fetch(urls.resetCalibration, {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrftoken,
            },
        })
        .then(response => response.json())
        .then(data => {
            console.log('Calibration reset:', data);
            
            currentStage = -1;
            
            const point = calibrationPoints[currentStage];
            document.getElementById('calibration-stage').textContent = 
                `Calibration Stage: ${currentStage} - ${point ? point.text : 'Unknown'}`;
            
            showCalibrationPoint(currentStage);
            updateProgressBar(currentStage);
            
            const statusBadge = document.querySelector('#calibration-status .badge');
            statusBadge.textContent = "Status: Not Calibrated";
            statusBadge.classList.remove('bg-success');
            statusBadge.classList.add('bg-secondary');
            
            cameraFeed.style.borderColor = 'red';
            startExamBtn.disabled = true;
            
            const progressBar = document.getElementById('calibration-progress');
            progressBar.classList.add('progress-bar-animated');
            progressBar.classList.remove('bg-success');
            
            alert('Calibration data has been reset. Press C to start calibration from stage -1.');
        })
        .catch(error => console.error('Error resetting calibration:', error));
    });


    // Handle start exam button click
    startExamBtn.addEventListener('click', function() {
        // Clear interval
        clearInterval(calibrationStatusInterval);
        
        // Release camera synchronously before navigating
        fetch(urls.releaseCamera, {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrftoken,
            },
            keepalive: true
        }).finally(() => {
            // Navigate after camera is released
            window.location.href = urls.startExamUrl;
        });
    });
});