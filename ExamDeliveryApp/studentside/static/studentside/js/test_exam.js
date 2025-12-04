document.addEventListener("DOMContentLoaded", function() {let eventLog = [];
    const status = document.getElementById("status");
    const eventLogInput = document.getElementById("event-log-input");
    const examForm = document.getElementById("exam-form");
    const warningOverlay = document.getElementById("warning-overlay");
    const mainContent = document.querySelector('.row');
    const examId = document.querySelector('.row').dataset.examId;

    let violationCount = 0;

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

    function blockPage(violationType) {
        violationCount++;
        eventLog.push({
            type: "security_violation", 
            violation: violationType, 
            ts: Date.now(),
            count: violationCount
        });

        // Show warning overlay
        warningOverlay.style.display = "block";
        
        // Blur and disable main content
        mainContent.classList.add("blocked");
        
        // Disable form submission
        examForm.querySelectorAll('input, button').forEach(el => {
            el.disabled = true;
        });

        // Submit violation log to server
        submitViolationLog();
    }

    function submitViolationLog() {
        // Send violation to server via AJAX
        fetch("{% url 'violation-check' %}", {
            method: "POST",
            headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'X-CSRFToken': csrftoken
            },
            body: `event_log=${encodeURIComponent(JSON.stringify(eventLog))}&violation=true&exam_id=${examId}`
        }).then(response => {
            // After logging violation, wait 3 seconds then redirect to dashboard
            setTimeout(function() {
            window.location.href = "/student";
            }, 800);
        });
    }

    // Load calibration and camera when page loads
    function reloadCameraWithCalibration() {
        fetch("/student/api/reload_camera_with_calibration/", {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrftoken,
            },
        })
        .then(response => response.json())
        .then(data => {
            console.log('Camera reload:', data);
            if (data.status === 'success') {
                console.log('Calibration loaded successfully');
                
                // Force reload the camera feed by updating the src with a cache-busting parameter
                const cameraFeed = document.getElementById('camera-feed');
                if (cameraFeed) {
                    const currentSrc = cameraFeed.src.split('?')[0]; // Remove existing query params
                    cameraFeed.src = currentSrc + '?t=' + new Date().getTime(); // Add timestamp to bust cache
                    
                    if (data.calibration_stage === 6) {
                        cameraFeed.style.borderColor = 'green';
                    }
                }
            } else if (data.status === 'warning') {
                console.warn(data.message);
                
                // Still reload camera feed even if no calibration data
                const cameraFeed = document.getElementById('camera-feed');
                if (cameraFeed) {
                    const currentSrc = cameraFeed.src.split('?')[0];
                    cameraFeed.src = currentSrc + '?t=' + new Date().getTime();
                }
            } else {
                console.error(data.message);
            }
        })
        .catch(error => console.error('Error reloading camera:', error));
    }

    // Call the function on page load
    try{reloadCameraWithCalibration();}catch(err){console.error('Error calling reloadCameraWithCalibration:', err);}
    


    // Log right-click attempts
    document.addEventListener("contextmenu", function(event) {
        event.preventDefault();
        eventLog.push({type: "keydown", key: "rightclick", ts: Date.now()});
    });

    // Block F12, Ctrl+Shift+I, Ctrl+Shift+J, Ctrl+Shift+C, Ctrl+U (common inspect shortcuts)
    document.addEventListener("keydown", function(event) {
        // F12
        if (event.key === "F12") {
            event.preventDefault();
            blockPage("F12_pressed");
            return false;
        }
        // Ctrl+Shift+I (Inspect)
        if (event.ctrlKey && event.shiftKey && event.key === "I") {
            event.preventDefault();
            blockPage("Ctrl+Shift+I");
            return false;
        }
        // Ctrl+Shift+J (Console)
        if (event.ctrlKey && event.shiftKey && event.key === "J") {
            event.preventDefault();
            blockPage("Ctrl+Shift+J");
            return false;
        }
        // Ctrl+Shift+C (Inspect Element)
        if (event.ctrlKey && event.shiftKey && event.key === "C") {
            event.preventDefault();
            blockPage("Ctrl+Shift+C");
            return false;
        }
        // Ctrl+U (View Source)
        if (event.ctrlKey && event.key.toLowerCase() === "u") {
            event.preventDefault();
            blockPage("Ctrl+U");
            return false;
        }
    });

    // Track keypresses (for legitimate typing)
    document.addEventListener("keydown", function(event) {
        // Skip if violation already occurred
        if (violationCount > 0) return;

        if (event.key === "Enter") {
            eventLog.push({type: "keydown", key: "Enter", ts: Date.now()});
        } else if (event.key === "Backspace") {
            eventLog.push({type: "keydown", key: "Backspace", ts: Date.now()});
        } else if (event.key === " ") {
            eventLog.push({type: "keydown", key: " ", ts: Date.now()});
        } else if (event.key.length === 1) {
            eventLog.push({type: "keydown", key: event.key, ts: Date.now()});
        }
    });

    // Track tab switches
    window.addEventListener("blur", function() {
        eventLog.push({type: "blur", ts: Date.now()});
        status.textContent = "Tab is unfocused - WARNING!";
        status.style.color = "red";
    });

    window.addEventListener("focus", function() {
        eventLog.push({type: "focus", ts: Date.now()});
        status.textContent = "Tab is focused";
        status.style.color = "darkblue";
    });

    // Track copy/paste
    document.addEventListener("copy", function() {
        eventLog.push({type: "copy", ts: Date.now()});
    });

    document.addEventListener("paste", function() {
        eventLog.push({type: "paste", ts: Date.now()});
    });

    // Before submitting, save the event log
    examForm.addEventListener("submit", function(event) {
        if (violationCount > 0) {
            event.preventDefault();
            alert("Cannot submit exam due to security violations.");
            return false;
        }
        eventLogInput.value = JSON.stringify(eventLog);
    });

    // Warn before leaving page AND release camera
    window.addEventListener("beforeunload", function(e) {
        // Release camera synchronously
        fetch("/student/api/release_camera/", {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrftoken,
            },
            keepalive: true
        });
        
        if (eventLog.length > 0 && violationCount === 0) {
            e.preventDefault();
            return "You have unsaved exam progress. Are you sure you want to leave?";
        }   
    });

    // Disable text selection to prevent easy copying
    document.addEventListener('selectstart', function(e) {
        if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
            e.preventDefault();
        }
    });
});
