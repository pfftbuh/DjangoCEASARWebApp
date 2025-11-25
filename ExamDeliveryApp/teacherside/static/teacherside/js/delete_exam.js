document.addEventListener('DOMContentLoaded', function() {

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

    document.addEventListener('click', function(e) {
        if (e.target.closest('.delete-exam-btn')) {
            const button = e.target.closest('.delete-exam-btn');
            const examId = button.dataset.examId;
            
            // Show confirmation dialog
            if (confirm('Are you sure you want to delete this exam? This action cannot be undone.')) {
                fetch(`/teacher/exam/${examId}/delete/`, {
                    method: 'POST',
                    headers: {
                        'X-CSRFToken': csrftoken,
                        'Content-Type': 'application/json'
                    },
                })
                .then(response => {
                    if (response.ok) {
                        // Redirect to manage exams page after successful deletion
                        window.location.href = '/teacher/exams/manage/';
                    } else {
                        alert('Error deleting exam. Please try again.');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred. Please try again.');
                });
            }
        }
    });
});