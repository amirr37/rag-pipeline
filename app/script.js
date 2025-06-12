 const retrieveBtn = document.getElementById('retrieveBtn');
    const generateBtn = document.getElementById('generateBtn');

    let lastRetrievedDocs = [];

    retrieveBtn.onclick = async () => {
        const query = document.getElementById('query').value.trim();
        const top_k = parseInt(document.getElementById('top_k').value);

        if (!query) return alert("Please enter a query.");

        const payload = {query, top_k};

        try {
            const res = await fetch('http://127.0.0.1:8000/retrieve', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });

            if (!res.ok) throw new Error(`Status ${res.status}`);

            const data = await res.json();
            lastRetrievedDocs = data.documents;

            document.getElementById('retrieveResponse').textContent = data.documents.join('\n\n---\n\n');
            document.getElementById('generateResponse').textContent = '';
        } catch (err) {
            document.getElementById('retrieveResponse').textContent = 'Error retrieving documents: ' + err.message;
        }
    };

    generateBtn.onclick = async () => {
        const query = document.getElementById('query').value.trim();

        // if (!query) return alert("Please enter a query.");
        // if (!lastRetrievedDocs.length) return alert("Please retrieve documents first.");

        // const context = lastRetrievedDocs.join('\n\n');
        const top_k = parseInt(document.getElementById('top_k').value);

        const payload = {query, top_k};

        try {
            const res = await fetch('http://127.0.0.1:8000/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });

            if (!res.ok) throw new Error(`Status ${res.status}`);

            const data = await res.json();
            document.getElementById('generateResponse').textContent = data.answer;
        } catch (err) {
            document.getElementById('generateResponse').textContent = 'Error generating answer: ' + err.message;
        }
    };
