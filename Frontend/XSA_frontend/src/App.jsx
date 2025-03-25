import { useState } from "react";
import Navbar from "./components/Navbar/Navbar";
import Upload from "./components/Upload/Upload";
import Result from "./components/Result/Result";
import "./main.css";

const App = () => {
  const [results, setResults] = useState([]);

  return (
    <div>
      <Navbar />
      <Upload onResult={setResults} />
      <Result results={results} />
    </div>
  );
};

export default App;
