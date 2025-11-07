document.addEventListener('DOMContentLoaded', () => {
    const interactButton = document.getElementById('interact-button');
    const interactionInput = document.getElementById('interaction-input');
    const significanceSlider = document.getElementById('significance-slider');
    const memoryVisualization = document.getElementById('memory-visualization');
    const explanationText = document.getElementById('explanation-text');

    const fetchMemoryState = async () => {
        const response = await fetch('/memory');
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
        const response = await fetch('/interact', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type: 'chat', content, significance: parseFloat(significance) }),
        });
        await response.json();
        fetchMemoryState();
        fetchExplanation();
    };

    const fetchExplanation = async () => {
        const response = await fetch('/explain');
        const data = await response.json();
        explanationText.textContent = data.explanation;
    };

    interactButton.addEventListener('click', interact);
    fetchMemoryState();
});
