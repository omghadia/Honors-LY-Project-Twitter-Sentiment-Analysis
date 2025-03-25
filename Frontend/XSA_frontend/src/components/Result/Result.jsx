import { useState } from "react";
import styles from "./styles.module.css";

const Result = ({ results }) => {
  const [visibleRows, setVisibleRows] = useState(5);

  if (!results || results.length === 0) {
    return <p className={styles.message}>No results to display.</p>;
  }

  // Check if the result is for a single text or a CSV file
  const isCSV =
    Array.isArray(results) &&
    results.length > 0 &&
    typeof results[0] === "object";

  return (
    <div className={styles.resultContainer}>
      <h2>Analysis Results</h2>

      {/* Display single text sentiment */}
      {!isCSV ? (
        <div className={styles.textResult}>
          <p>
            <strong>Text:</strong> {results.text}
          </p>
          <p>
            <strong>Sentiment:</strong> {results.sentiment}
          </p>
        </div>
      ) : (
        <>
          <table className={styles.table}>
            <thead>
              <tr>
                {Object.keys(results[0]).map((key) => (
                  <th key={key}>{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {results.slice(0, visibleRows).map((row, index) => (
                <tr key={index}>
                  {Object.values(row).map((value, i) => (
                    <td key={i}>{value}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>

          {/* Show "Load More" button if more than 5 rows */}
          {results.length > visibleRows && (
            <button
              className={styles.loadMoreButton}
              onClick={() => setVisibleRows((prev) => prev + 5)}
            >
              Load More
            </button>
          )}
        </>
      )}
    </div>
  );
};

export default Result;
