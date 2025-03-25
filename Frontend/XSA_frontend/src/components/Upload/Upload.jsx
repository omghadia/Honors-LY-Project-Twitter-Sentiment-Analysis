import { useState } from "react";
import styles from "./styles.module.css";

const Upload = ({ onResult }) => {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleTextSubmit = async () => {
    if (!text.trim()) return alert("Please enter some text.");

    setLoading(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/predict_text/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const result = await response.json();
      onResult([result]); // Send the result to the parent component
    } catch (error) {
      console.error("Error:", error);
    }
    setLoading(false);
  };

  const handleFileSubmit = async () => {
    if (!file) return alert("Please upload a CSV file.");

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict_csv/", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      onResult(result);
    } catch (error) {
      console.error("Error:", error);
    }
    setLoading(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = () => {
    setDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const droppedFile = e.dataTransfer.files[0];

    if (droppedFile && droppedFile.type === "text/csv") {
      setFile(droppedFile);
    } else {
      alert("Please upload a valid CSV file.");
    }
  };

  return (
    <div className={styles.uploadContainer}>
      <h2>Enter Text for Sentiment Analysis</h2>
      <textarea
        className={styles.textarea}
        rows="4"
        placeholder="Enter text here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button
        className={styles.button}
        onClick={handleTextSubmit}
        disabled={loading}
      >
        {loading ? "Analyzing..." : "Analyze Text"}
      </button>

      <h2>Or Upload CSV File</h2>
      <div
        className={`${styles.dropZone} ${dragging ? styles.dragging : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {file ? (
          <p>{file.name}</p>
        ) : (
          <p>Drag & Drop your CSV file here or click to upload</p>
        )}
      </div>
      <input
        type="file"
        accept=".csv"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button
        className={styles.button}
        onClick={handleFileSubmit}
        disabled={loading}
      >
        {loading ? "Uploading..." : "Upload & Analyze CSV"}
      </button>
    </div>
  );
};

export default Upload;
