import { useState } from 'react';
import './App.css';

function App() {
  const [url, setUrl] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  const handleUrlSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch('/api/webhook', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      });
      const data = await response.json();
      console.log(data);
    } catch (error) {
      console.error('Error submitting URL:', error);
    }
    setLoading(false);
  };

  const handleQuestionSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });
      const data = await response.json();
      setAnswer(data.answer);
    } catch (error) {
      console.error('Error submitting question:', error);
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Bajaj Finserv AI Assistant</h1>
      </header>
      <main>
        <section>
          <h2>Submit a document URL</h2>
          <form onSubmit={handleUrlSubmit}>
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Enter document URL"
              required
            />
            <button type="submit" disabled={loading}>
              {loading ? 'Processing...' : 'Submit'}
            </button>
          </form>
        </section>
        <section>
          <h2>Ask a question</h2>
          <form onSubmit={handleQuestionSubmit}>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Enter your question"
              required
            />
            <button type="submit" disabled={loading}>
              {loading ? 'Thinking...' : 'Ask'}
            </button>
          </form>
        </section>
        {answer && (
          <section>
            <h2>Answer</h2>
            <p>{answer}</p>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
