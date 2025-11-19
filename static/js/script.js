document.getElementById('ner-form').addEventListener('submit', function (event) {
    event.preventDefault();

    const text = document.getElementById('input-text').value;

    fetch('/predict', {
        method: 'POST',
        body: new URLSearchParams({ text: text }),
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    })
    .then(response => response.json())
    .then(data => {
        const entities = data.entities;
        const entitySentence = data.entity_sentence; // 获取返回的完整句子

        const entityList = document.getElementById('entity-list');
        entityList.innerHTML = '';

        // 显示整句话与每个字符的标注
        let formattedSentence = "<strong>句子标注：</strong><br>";
        formattedSentence += entitySentence.replace(/ /g, " <br>"); // 每个词标注换行显示
        entityList.innerHTML = formattedSentence;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
