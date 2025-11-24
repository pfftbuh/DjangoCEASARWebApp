document.addEventListener('DOMContentLoaded', function() {
    const accessCodeCheckbox = document.getElementById('access_code_required');
    const accessCodeInput = document.getElementById('access_code');
    
    // Handle checkbox change
    accessCodeCheckbox.addEventListener('change', function() {
        accessCodeInput.disabled = !this.checked;
        if (!this.checked) {
            accessCodeInput.value = '';
        }
    });
    
    // Initialize the state on page load
    accessCodeInput.disabled = !accessCodeCheckbox.checked;
});