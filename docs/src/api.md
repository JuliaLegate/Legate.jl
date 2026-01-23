# Public API
```@raw html
<AuthorBadge
  author="krasow"
  platform="github"
/>
<AuthorBadge
  author="ejmeitz"
  platform="github"
/>
```

User facing functions supported by Legate.jl. The cpp wrapper API is located
```@raw html
<a href="./CppAPI/index.html" target="_self">here</a>.
```

```@contents
Pages = ["api.md"]
Depth = 2:2
```

## Runtime Control and Performance Timing
```@autodocs
Modules = [Legate]
Pages = ["api/runtime.jl"]
```

## Data Structures and Storage Management
```@autodocs
Modules = [Legate]
Pages = ["api/data.jl"]
```

## Task Creation and Execution
```@autodocs
Modules = [Legate]
Pages = ["api/task.jl"]
```

## Core Types and Interfaces 
```@autodocs
Modules = [Legate]
Pages = ["api/types.jl"]
```