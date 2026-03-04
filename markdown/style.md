# Coding Style Guide: GNN Scalability Repository

> **Purpose:** This document defines the architectural principles, design patterns, and coding conventions used in this repository. Use this as a reference when extending the codebase or instructing LLMs to generate new code.

---

## 🏛️ Core Architecture Philosophy

### **"Clean Architecture + Design Patterns"**

This repository follows **Clean Architecture** principles combined with classic **Gang of Four design patterns**. Every component is:

1. **Testable** - Can be mocked and unit tested in isolation
2. **Extensible** - New features added without modifying existing code (Open/Closed)
3. **Modular** - Clear boundaries between layers
4. **Type-safe** - Uses type hints everywhere
5. **Well-documented** - Docstrings for all public APIs

---

## 🎯 SOLID Principles (MANDATORY)

### **Every piece of code MUST follow all 5 SOLID principles:**

#### 1. **Single Responsibility Principle (SRP)**
- ✅ Each class/function has ONE reason to change
- ✅ No God classes (classes that do everything)
- ✅ Separate concerns (loading ≠ processing ≠ displaying)

**Example:**
```python
# ✅ GOOD - Single responsibility
class DatasetLoader:
    def load(self, path: str) -> Data:
        """Only loads data."""
        pass

class DatasetValidator:
    def validate(self, data: Data) -> bool:
        """Only validates data."""
        pass

# ❌ BAD - Multiple responsibilities
class DatasetManager:
    def load_and_validate_and_process(self, path: str):
        """Does too many things!"""
        pass
```

#### 2. **Open/Closed Principle (OCP)**
- ✅ Open for extension, closed for modification
- ✅ Use inheritance/composition to extend behavior
- ✅ Never use if/elif chains for polymorphic behavior

**Example:**
```python
# ✅ GOOD - Extend by adding new class
class NewBackend(GraphBackend):
    def materialize_exact(self):
        # New implementation
        pass

# ❌ BAD - Modify existing code
def process(backend_type):
    if backend_type == 'python':
        # ...
    elif backend_type == 'cpp':
        # ...
    elif backend_type == 'new':  # ← Modifying existing function!
        # ...
```

#### 3. **Liskov Substitution Principle (LSP)**
- ✅ Subtypes must be substitutable for base types
- ✅ All implementations follow the same contract
- ✅ No surprising behavior in subclasses

**Example:**
```python
# ✅ GOOD - Both can substitute GraphBackend
backend: GraphBackend = PythonBackend()  # Works
backend: GraphBackend = CppBackend()     # Also works
result = backend.materialize_exact()      # Same interface

# ❌ BAD - Subclass breaks contract
class BrokenBackend(GraphBackend):
    def materialize_exact(self):
        raise NotImplementedError("Oops!")  # ← Breaks LSP!
```

#### 4. **Interface Segregation Principle (ISP)**
- ✅ Small, focused interfaces
- ✅ Clients shouldn't depend on methods they don't use
- ✅ Prefer multiple specific interfaces over one fat interface

**Example:**
```python
# ✅ GOOD - Focused interfaces
class Readable(ABC):
    @abstractmethod
    def read(self) -> Data: pass

class Writable(ABC):
    @abstractmethod
    def write(self, data: Data) -> None: pass

# ❌ BAD - Fat interface
class DataManager(ABC):
    @abstractmethod
    def read(self): pass
    @abstractmethod
    def write(self): pass
    @abstractmethod
    def compress(self): pass
    @abstractmethod
    def encrypt(self): pass  # ← Most clients don't need this!
```

#### 5. **Dependency Inversion Principle (DIP)**
- ✅ Depend on abstractions, not concretions
- ✅ High-level modules independent of low-level modules
- ✅ Use Dependency Injection

**Example:**
```python
# ✅ GOOD - Depends on abstraction
class BenchmarkCommand:
    def __init__(self, backend: GraphBackend):  # ← Abstract type
        self.backend = backend

# ❌ BAD - Depends on concrete class
class BenchmarkCommand:
    def __init__(self):
        self.backend = PythonBackend()  # ← Concrete! Tightly coupled!
```

---

## 🎨 Design Patterns (By Use Case)

### **When to Use Each Pattern**

#### **Strategy Pattern** - For Interchangeable Algorithms
**Use when:** You have multiple ways to do the same thing

**Example in this repo:**
- `GraphBackend` (abstract) → `PythonBackend`, `CppBackend`
- Different backends, same interface

**Template:**
```python
class StrategyInterface(ABC):
    @abstractmethod
    def execute(self) -> Result:
        pass

class ConcreteStrategyA(StrategyInterface):
    def execute(self) -> Result:
        # Implementation A
        pass

class ConcreteStrategyB(StrategyInterface):
    def execute(self) -> Result:
        # Implementation B
        pass
```

#### **Factory Pattern** - For Object Creation
**Use when:** Creation logic is complex or varies

**Example in this repo:**
- `BackendFactory.create('python')`
- `DatasetFactory.get_data('HGB', 'DBLP', 'author')`

**Template:**
```python
class Factory:
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, impl_class: Type):
        cls._registry[name] = impl_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseType:
        if name not in cls._registry:
            raise ValueError(f"Unknown type: {name}")
        return cls._registry[name](**kwargs)
```

#### **Template Method Pattern** - For Common Workflow
**Use when:** Multiple classes share a workflow but differ in details

**Example in this repo:**
- `BaseGraphLoader` (defines workflow) → `HGBLoader`, `OGBLoader`

**Template:**
```python
class AbstractClass(ABC):
    def template_method(self):
        """Defines the workflow (DON'T OVERRIDE)"""
        self._step1()
        self._step2()
        self._step3()
    
    @abstractmethod
    def _step1(self): pass  # Subclasses implement
    
    @abstractmethod
    def _step2(self): pass
```

#### **Command Pattern** - For Operations
**Use when:** You want to encapsulate operations as objects

**Example in this repo:**
- `BenchmarkCommand`, `MiningCommand`, `ListCommand`

**Template:**
```python
class Command(ABC):
    @abstractmethod
    def execute(self, args) -> None:
        pass

class ConcreteCommand(Command):
    def execute(self, args) -> None:
        # Do the operation
        pass
```

#### **Adapter Pattern** - For Interface Conversion
**Use when:** You need to make incompatible interfaces work together

**Example in this repo:**
- `PyGToCppAdapter` (converts PyG format → C++ format)

**Template:**
```python
class Target(ABC):
    @abstractmethod
    def request(self) -> str:
        pass

class Adapter(Target):
    def __init__(self, adaptee: Adaptee):
        self.adaptee = adaptee
    
    def request(self) -> str:
        # Convert adaptee's interface to Target's interface
        return self.adaptee.specific_request().upper()
```

#### **Singleton Pattern** - For Global State
**Use when:** Exactly one instance should exist (use sparingly!)

**Example in this repo:**
- `Config` class (single source of configuration)

**Template:**
```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

---

## 📁 File Organization Rules

### **Module Structure Philosophy**

```
src/
├── data/           # Input data handling (Factory pattern)
│   ├── base.py     # Abstract interface
│   ├── loaders.py  # Concrete implementations
│   └── factory.py  # Creation logic
├── kernels/        # Core algorithms (Strategy pattern)
│   ├── exact.py    # Algorithm implementation
│   └── kmv.py
├── backend/        # Execution engines (Strategy pattern)
│   ├── base.py     # Abstract interface
│   ├── python_backend.py
│   └── cpp_backend.py
├── bridge/         # External tool adapters (Adapter pattern)
│   ├── cpp_adapter.py
│   └── anyburl.py
├── models.py       # GNN architectures
├── config.py       # Configuration (Singleton)
└── utils.py        # Pure functions
```

### **File Organization Rules**

1. **One pattern per module** - Don't mix factories and strategies in the same file
2. **Abstract base classes in `base.py`** - Concrete implementations in other files
3. **Factory in separate file** - Don't mix creation logic with business logic
4. **Maximum 400 lines per file** - If longer, split into multiple files
5. **Related classes together** - But separated by blank lines with comments

---

## 📝 Code Conventions

### **Naming Conventions**

| Type | Convention | Example |
|------|-----------|---------|
| Classes | PascalCase | `GraphBackend`, `DatasetLoader` |
| Functions/Methods | snake_case | `load_data()`, `get_result()` |
| Constants | UPPER_SNAKE | `MAX_SIZE`, `DEFAULT_TIMEOUT` |
| Private methods | _leading_underscore | `_internal_helper()` |
| Type aliases | PascalCase | `EdgeType = Tuple[str, str, str]` |
| Modules | snake_case | `python_backend.py` |

### **Type Hints (MANDATORY)**

```python
# ✅ GOOD - Type hints everywhere
def process_graph(g: HeteroData, 
                  metapath: List[Tuple[str, str, str]],
                  config: Dict[str, Any]) -> Data:
    """Process heterogeneous graph."""
    result: Data = ...
    return result

# ❌ BAD - No type hints
def process_graph(g, metapath, config):
    result = ...
    return result
```

### **Docstring Format (Google Style)**

```python
def materialize_exact(g_hetero: HeteroData,
                     metapath: List[Tuple[str, str, str]],
                     target_ntype: str) -> Tuple[Data, float]:
    """
    Materializes a homogeneous graph from a metapath.
    
    Uses sparse matrix multiplication to compute the exact
    connectivity following the specified metapath.
    
    Args:
        g_hetero: Input heterogeneous graph
        metapath: List of edge type tuples defining the path
        target_ntype: Target node type for the output graph
        
    Returns:
        Tuple of (materialized graph, computation time in seconds)
        
    Raises:
        ValueError: If metapath is empty or invalid
        
    Example:
        >>> g = load_graph('DBLP')
        >>> path = [('author', 'writes', 'paper'), ('paper', 'cites', 'paper')]
        >>> result, time = materialize_exact(g, path, 'paper')
        >>> print(f"Graph with {result.num_edges} edges in {time:.2f}s")
    """
    pass
```

### **Import Organization**

```python
# 1. Standard library imports (alphabetical)
import os
import sys
from typing import Dict, List, Tuple

# 2. Third-party imports (alphabetical)
import torch
import pandas as pd
from torch_geometric.data import Data, HeteroData

# 3. Local imports (alphabetical)
from .base import GraphBackend
from ..kernels import ExactMaterializationKernel
from ..config import config
```

### **Error Handling**

```python
# ✅ GOOD - Specific exceptions with context
def load_file(path: str) -> Data:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty CSV file: {path}")
    
    return process(data)

# ❌ BAD - Bare except, no context
def load_file(path: str):
    try:
        data = pd.read_csv(path)
    except:  # ← Don't catch everything!
        return None  # ← Lost error information!
```

---

## 🚫 Anti-Patterns to AVOID

### **1. God Classes**
```python
# ❌ BAD - Does everything
class DataManager:
    def load(self): pass
    def validate(self): pass
    def transform(self): pass
    def save(self): pass
    def visualize(self): pass
    def train_model(self): pass  # ← Why is this here?!
```

### **2. If/Elif for Polymorphism**
```python
# ❌ BAD - Violates Open/Closed
def process(backend_type: str):
    if backend_type == 'python':
        return PythonBackend().run()
    elif backend_type == 'cpp':
        return CppBackend().run()
    # ← Adding new backend requires modifying this function!

# ✅ GOOD - Use polymorphism
backend: GraphBackend = BackendFactory.create(backend_type)
return backend.run()
```

### **3. Tight Coupling**
```python
# ❌ BAD - Tightly coupled to concrete class
class Benchmark:
    def __init__(self):
        self.backend = PythonBackend()  # ← Can't swap!

# ✅ GOOD - Loose coupling via interface
class Benchmark:
    def __init__(self, backend: GraphBackend):
        self.backend = backend  # ← Can inject any backend!
```

### **4. Magic Numbers/Strings**
```python
# ❌ BAD
if status == 1:  # What does 1 mean?
    timeout = 3600  # Why 3600?

# ✅ GOOD
STATUS_SUCCESS = 1
DEFAULT_TIMEOUT_SECONDS = 3600

if status == STATUS_SUCCESS:
    timeout = DEFAULT_TIMEOUT_SECONDS
```

### **5. Mutable Default Arguments**
```python
# ❌ BAD - Mutable default
def process(items: List[str] = []):  # ← Bug!
    items.append('new')
    return items

# ✅ GOOD - Use None
def process(items: Optional[List[str]] = None) -> List[str]:
    if items is None:
        items = []
    items.append('new')
    return items
```

---

## ✅ Code Quality Checklist

**Before committing code, verify:**

- [ ] **SOLID:** Does this follow all 5 SOLID principles?
- [ ] **Patterns:** Is there an appropriate design pattern being used?
- [ ] **Types:** Are there type hints on all function signatures?
- [ ] **Docs:** Do public APIs have docstrings?
- [ ] **Tests:** Can this be easily unit tested?
- [ ] **DRY:** Is there any code duplication?
- [ ] **Naming:** Are names clear and follow conventions?
- [ ] **Errors:** Are exceptions specific and informative?
- [ ] **Imports:** Are imports organized correctly?
- [ ] **Length:** Is any file over 400 lines? (Split if yes)

---

## 🎯 Extending the System: Step-by-Step

### **Adding a New Backend**

1. **Create file:** `src/backend/my_backend.py`
2. **Inherit:** `class MyBackend(GraphBackend):`
3. **Implement:** All abstract methods (5 methods)
4. **Register:** `BackendFactory.register('mybackend', MyBackend)`
5. **Test:** `python main.py benchmark --backend mybackend`

**DO NOT:**
- ❌ Modify existing backend files
- ❌ Add if/elif to main.py
- ❌ Change GraphBackend interface (unless absolutely necessary)

### **Adding a New Dataset Source**

1. **Create class:** In `src/data/loaders.py`
2. **Inherit:** `class MyLoader(BaseGraphLoader):`
3. **Implement:** `load()` method
4. **Register:** `DatasetFactory.register_loader('mysource', MyLoader)`
5. **Test:** Load a dataset using your loader

**DO NOT:**
- ❌ Modify BaseGraphLoader (unless adding common functionality)
- ❌ Duplicate code from other loaders (extract to base class)

### **Adding a New Command**

1. **Create class:** `class MyCommand:` in `main.py`
2. **Implement:** `execute(self, args)` method
3. **Add parser:** Create subparser in `create_parser()`
4. **Route:** Add routing in `main()` function

---

## 💡 Philosophy and Best Practices

### **"Composition Over Inheritance"**
Prefer composing objects over extending classes.

```python
# ✅ GOOD - Composition
class Benchmark:
    def __init__(self, backend: GraphBackend, logger: Logger):
        self.backend = backend
        self.logger = logger

# ⚠️ OKAY - Inheritance (only for IS-A relationships)
class PythonBackend(GraphBackend):  # PythonBackend IS-A GraphBackend
    pass
```

### **"Fail Fast"**
Catch errors early with validation.

```python
def process(k: int):
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    # ... rest of code
```

### **"Explicit is Better Than Implicit"**
Make intentions clear.

```python
# ✅ GOOD - Clear intent
result = backend.materialize_exact()

# ❌ BAD - What does this do?
result = backend.run()
```

### **"Don't Repeat Yourself (DRY)"**
Extract common logic.

```python
# ✅ GOOD - Common logic in base class
class BaseGraphLoader(ABC):
    def _ensure_features(self, g, ntype):
        # Common feature generation logic
        pass

# ❌ BAD - Duplicated in every loader
class HGBLoader:
    def load(self):
        # Feature generation code
        pass

class OGBLoader:
    def load(self):
        # Same feature generation code (duplicated!)
        pass
```

---

## 🧪 Testing Guidelines

### **Unit Test Structure**

```python
# tests/test_backends.py
import pytest
from src.backend import BackendFactory, GraphBackend

class MockBackend(GraphBackend):
    """Mock backend for testing."""
    # ... implementation

def test_backend_factory_creation():
    """Test factory can create backends."""
    BackendFactory.register('mock', MockBackend)
    backend = BackendFactory.create('mock')
    assert isinstance(backend, GraphBackend)

def test_backend_substitutability():
    """Test Liskov Substitution Principle."""
    for backend_name in ['python', 'cpp', 'mock']:
        backend = BackendFactory.create(backend_name)
        # All backends should support same interface
        assert hasattr(backend, 'materialize_exact')
        assert hasattr(backend, 'materialize_kmv')
```

---

## 📚 Recommended Reading

1. **"Design Patterns"** by Gang of Four - Original design patterns book
2. **"Clean Code"** by Robert C. Martin - Code quality principles
3. **"Refactoring"** by Martin Fowler - How to improve existing code
4. **PEP 8** - Python style guide

---

## 🤖 Instructions for LLMs

When extending this codebase:

1. **Read this document first** to understand architecture
2. **Follow SOLID principles** - All 5, no exceptions
3. **Use appropriate design patterns** - Don't reinvent the wheel
4. **Match existing code style** - Consistency is key
5. **Add type hints** - Always
6. **Write docstrings** - For public APIs
7. **Avoid anti-patterns** - Listed above
8. **Test extensibility** - Can you add features without modifying existing code?

### **LLM Prompt Template**

When asking an LLM to extend this code:

```
This repository follows SOLID principles and uses design patterns 
(see CODING_STYLE.md). When adding [FEATURE], please:

1. Identify which design pattern to use (Strategy/Factory/Command/etc.)
2. Follow the existing pattern in src/[MODULE]/
3. Ensure SOLID compliance (especially Open/Closed Principle)
4. Add type hints and docstrings
5. Do not modify existing classes unless absolutely necessary

Example: To add a new backend, inherit from GraphBackend and 
register with BackendFactory. Do NOT add if/elif statements.
```

---

## 🎓 Summary

**This repository prioritizes:**
- ✅ **SOLID principles** above all else
- ✅ **Design patterns** for common problems
- ✅ **Type safety** with hints everywhere
- ✅ **Extensibility** without modification
- ✅ **Testability** via dependency injection
- ✅ **Clarity** over cleverness

**Remember:** Good code is code that's easy to delete and replace, not hard to understand and modify.

---

**Version:** 2.0  
**Last Updated:** 2025
**Maintainer:** gilchrsin 