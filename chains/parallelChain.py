from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = GoogleGenerativeAI(model="gemini-3.1-flash-lite")

notesPrompt = PromptTemplate(
    template="Create brief notes from below text \n {topic}",
    input_variables=["topic"]
)

quizzPrompt = PromptTemplate(
    template="Generate 5 questions from below topic \n {topic}",
    input_variables=["topic"]
)

mergePrompt = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallelChain = RunnableParallel({
    "notes":notesPrompt | model | parser,
    "quiz":quizzPrompt | model | parser
})

merger_chain = mergePrompt | model | parser

chain = parallelChain | merger_chain

result = chain.invoke({"topic":"""

# React `useEffect` Hook

## Definition
`useEffect` is a React Hook used to perform **side effects** in functional components after rendering.

### Common Side Effects
- Fetch API data
- Add/remove event listeners
- Start/stop timers
- Update document title
- Access localStorage
- WebSocket connections

---

## Syntax

```jsx
useEffect(() => {
  // Side effect

  return () => {
    // Cleanup (optional)
  };
}, [dependencies]);
```

---

## 1. No Dependency Array

Runs after **every render**.

```jsx
useEffect(() => {
  console.log("Runs after every render");
});
```

---

## 2. Empty Dependency Array

Runs **only once** after the initial render.

```jsx
useEffect(() => {
  console.log("Runs once");
}, []);
```

---

## 3. With Dependencies

Runs on the initial render and whenever the dependency changes.

```jsx
useEffect(() => {
  console.log(count);
}, [count]);
```

---

## Cleanup Function

Used to prevent memory leaks by cleaning up resources.

```jsx
useEffect(() => {
  const id = setInterval(() => {
    console.log("Running...");
  }, 1000);

  return () => clearInterval(id);
}, []);
```

---

## Example: Fetch API

```jsx
useEffect(() => {
  async function fetchUsers() {
    const res = await fetch("https://jsonplaceholder.typicode.com/users");
    const data = await res.json();
    setUsers(data);
  }

  fetchUsers();
}, []);
```

---

## Best Practices

- Include all required dependencies.
- Clean up timers, event listeners, and subscriptions.
- Keep each `useEffect` focused on a single responsibility.
- Avoid updating a dependency inside the same effect unless necessary.

---

## Quick Summary

- Performs side effects after rendering.
- No dependency array → Every render.
- Empty array (`[]`) → Runs once.
- Dependency array (`[count]`) → Runs when dependencies change.
- Cleanup function → Prevents memory leaks.
"""})

print(result)