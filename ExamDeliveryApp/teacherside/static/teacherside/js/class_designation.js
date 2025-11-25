document.addEventListener('DOMContentLoaded', function() {
    const classList = document.getElementById('class_list');
    const addClassBtn = document.getElementById('add_class_btn');
    const selectedClassesList = document.getElementById('selected_classes_list');
    const selectedClassesInput = document.getElementById('selected_classes');
    let selectedClasses = [];

    addClassBtn.addEventListener('click', function() {
        const selectedOption = classList.options[classList.selectedIndex];
        const classValue = selectedOption.value;
        if (classValue && !selectedClasses.includes(classValue)) {
            selectedClasses.push(classValue);
            updateSelectedClasses();
        }
    });

    // Populate selectedClasses with existing class designations if present
    if (selectedClassesInput.value) {
        selectedClasses = selectedClassesInput.value.split(',').filter(Boolean);
        updateSelectedClasses();
    }

    function updateSelectedClasses() {
        selectedClassesList.innerHTML = '';
        selectedClasses.forEach(function(className, idx) {
            const li = document.createElement('li');
            li.className = 'list-group-item d-flex justify-content-between align-items-center';
            li.textContent = className;
            const removeBtn = document.createElement('button');
            removeBtn.type = 'button';
            removeBtn.className = 'btn btn-danger btn-sm';
            removeBtn.textContent = 'Remove';
            removeBtn.onclick = function() {
                selectedClasses.splice(idx, 1);
                updateSelectedClasses();
            };
            li.appendChild(removeBtn);
            selectedClassesList.appendChild(li);
        });
        selectedClassesInput.value = selectedClasses.join(',');
    }
});