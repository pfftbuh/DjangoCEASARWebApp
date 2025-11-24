document.addEventListener('DOMContentLoaded', function() {
    const choicesContainer = document.getElementById('choices_container');
    const answerContainer = document.getElementById('answer_container');
    const questionType = document.getElementById('question_type');
    const answerInput = document.getElementById('correct_answer');

    questionType.addEventListener('change', function() {
        choicesContainer.innerHTML = '';
        answerContainer.innerHTML = '';

        if (this.value === 'mcq_one') {
            // MCQ One Answer - Radio buttons for correct answer
            let html = `
                <div class="form-group mt-3">
                    <label><strong>Choices</strong></label>
                    <div id="mcq_choices">
                        <div class="input-group mb-2">
                            <input type="text" name="choices" class="form-control choice-input" placeholder="Choice 1" required>
                            <button type="button" class="btn btn-outline-secondary add-choice">+</button>
                        </div>
                    </div>
                </div>
            `;
            choicesContainer.innerHTML = html;

            answerContainer.innerHTML = `
                <label><strong>Correct Answer</strong></label>
                <div id="radio_answers"></div>
            `;

            updateRadioAnswers();

        } else if (this.value === 'mcq_multi') {
            // MCQ Multiple Answers - Checkboxes for correct answers
            let html = `
                <div class="form-group mt-3">
                    <label><strong>Choices</strong></label>
                    <div id="mcq_choices">
                        <div class="input-group mb-2">
                            <input type="text" name="choices" class="form-control choice-input" placeholder="Choice 1" required>
                            <button type="button" class="btn btn-outline-secondary add-choice">+</button>
                        </div>
                    </div>
                </div>
            `;
            choicesContainer.innerHTML = html;

            answerContainer.innerHTML = `
                <label><strong>Correct Answer(s)</strong></label>
                <div id="checkbox_answers"></div>
            `;

            updateCheckboxAnswers();

        } else if (this.value === 'true_false') {
            // True/False - Radio buttons
            choicesContainer.innerHTML = `
                <input type="hidden" name="choices" value="True">
                <input type="hidden" name="choices" value="False">
            `;

            answerContainer.innerHTML = `
                <label><strong>Correct Answer</strong></label>
                <div>
                    <div class="form-check">
                        <input class="form-check-input" type="radio" name="correct_answer" id="correct_answer_true" value="True" required>
                        <label class="form-check-label" for="correct_answer_true">True</label>
                    </div>
                    <div class="form-check">
                        <input class="form-check-input" type="radio" name="correct_answer" id="correct_answer_false" value="False" required>
                        <label class="form-check-label" for="correct_answer_false">False</label>
                    </div>
                </div>
            `;

        } else if (this.value === 'numerical') {
            // Numerical - Simple input
            answerContainer.innerHTML = `
                <label for="correct_answer"><strong>Correct Answer</strong></label>
                <input type="text" class="form-control" id="correct_answer" name="correct_answer" placeholder="Enter numerical answer" required>
            `;
        }
    });

    // Function to update radio buttons for MCQ One
    function updateRadioAnswers() {
        const choices = document.querySelectorAll('.choice-input');
        const radioContainer = document.getElementById('radio_answers');
        if (!radioContainer) return;

        radioContainer.innerHTML = '';
        choices.forEach((choice, index) => {
            if (choice.value.trim()) {
                const div = document.createElement('div');
                div.className = 'form-check';
                div.innerHTML = `
                    <input class="form-check-input" type="radio" name="correct_answer" id="radio_${index}" value="${choice.value}" required>
                    <label class="form-check-label" for="radio_${index}">${choice.value}</label>
                `;
                radioContainer.appendChild(div);
            }
        });
    }

    // Function to update checkboxes for MCQ Multi
    function updateCheckboxAnswers() {
        const choices = document.querySelectorAll('.choice-input');
        const checkboxContainer = document.getElementById('checkbox_answers');
        if (!checkboxContainer) return;

        checkboxContainer.innerHTML = '';
        choices.forEach((choice, index) => {
            if (choice.value.trim()) {
                const div = document.createElement('div');
                div.className = 'form-check';
                div.innerHTML = `
                    <input class="form-check-input" type="checkbox" name="correct_answer" id="checkbox_${index}" value="${choice.value}">
                    <label class="form-check-label" for="checkbox_${index}">${choice.value}</label>
                `;
                checkboxContainer.appendChild(div);
            }
        });
    }

    // Add dynamic choice fields and update answer options
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('add-choice')) {
            const mcqChoices = document.getElementById('mcq_choices');
            const count = mcqChoices.querySelectorAll('input').length + 1;
            const div = document.createElement('div');
            div.className = 'input-group mb-2';
            div.innerHTML = `
                <input type="text" name="choices" class="form-control choice-input" placeholder="Choice ${count}" required>
                <button type="button" class="btn btn-outline-danger remove-choice">-</button>
            `;
            mcqChoices.appendChild(div);

            // Update answer options
            if (questionType.value === 'mcq_one') {
                updateRadioAnswers();
            } else if (questionType.value === 'mcq_multi') {
                updateCheckboxAnswers();
            }
        }

        if (e.target.classList.contains('remove-choice')) {
            e.target.parentElement.remove();

            // Update answer options
            if (questionType.value === 'mcq_one') {
                updateRadioAnswers();
            } else if (questionType.value === 'mcq_multi') {
                updateCheckboxAnswers();
            }
        }
    });

    // Update answer options when choices are typed
    document.addEventListener('input', function(e) {
        if (e.target.classList.contains('choice-input')) {
            if (questionType.value === 'mcq_one') {
                updateRadioAnswers();
            } else if (questionType.value === 'mcq_multi') {
                updateCheckboxAnswers();
            }
        }
    });
});