document.addEventListener('DOMContentLoaded', () => {
    const loginButton = document.getElementById('login-button');
    const usernameInput = document.getElementById('username-input');
    const interactButton = document.getElementById('interact-button');
    const trainButton = document.getElementById('train-button');
    const interactionInput = document.getElementById('interaction-input');
    const significanceSlider = document.getElementById('significance-slider');
    const memoryVisualization = document.getElementById('memory-visualization');
    const explanationText = document.getElementById('explanation-text');

    let token = null;

    const login = async () => {
        const username = usernameInput.value;
        const response = await fetch('/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username }),
        });
        const data = await response.json();
        token = data.token;
        if (token) {
            document.querySelector('.login-container').style.display = 'none';
            document.querySelector('.interaction-container').style.display = 'block';
            fetchMemoryState();
        }
    };

    const fetchMemoryState = async () => {
        const response = await fetch('/memory', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await response.json();
        renderMemory(data.memory_state[0]);
    };

    const renderMemory = (memoryState) => {
        memoryVisualization.innerHTML = '';
        memoryState.forEach(value => {
            const cell = document.createElement('div');
            cell.classList.add('memory-cell');
            cell.style.backgroundColor = `rgba(0, 0, 255, ${Math.abs(value)})`;
            memoryVisualization.appendChild(cell);
        });
    };

    const interact = async () => {
        const content = interactionInput.value;
        const significance = significanceSlider.value;
        await fetch('/interact', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ type: 'chat', content, significance: parseFloat(significance) }),
        });
        fetchMemoryState();
        fetchExplanation();
    };

    const train = async () => {
        await fetch('/train', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}` }
        });
    };

    const fetchExplanation = async () => {
        const response = await fetch('/explain', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await response.json();
        explanationText.textContent = data.explanation;
    };

    loginButton.addEventListener('click', login);
    interactButton.addEventListener('click', interact);
    trainButton.addEventListener('click', train);
});
