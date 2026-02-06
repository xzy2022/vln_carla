# Hexagonal + DDD Architecture Standard (v2.1)

## Purpose
Define **strict, production-grade rules** for building systems using  
**Hexagonal Architecture + DDD + Clean layering**.

This is an **implementation standard**, not a conceptual discussion.  
All dependency, ownership, and placement rules are **hard constraints**.

---

## 1. Layered Structure (Default)

```text
- domain/            # Core business model: Entities, Value Objects, Aggregates, Domain Services
- usecases/          # Application layer: Use Cases (Application Services), orchestration, app-level ports
- adapters/          # Primary (driving) adapters: HTTP, GraphQL, CLI, MQ consumers, presenters
- infrastructure/    # Secondary (driven) technical adapters: DB, cache, MQ producer, external clients
- app/               # Composition Root only: DI wiring, config, bootstrap
```

Notes:
- In Hexagonal terminology, both `adapters/` and `infrastructure/` are adapters.
- This split is intentional:
  - `adapters/` → inbound protocols & presentation
  - `infrastructure/` → outbound technical integrations

---

## 2. Dependency Rules (Hard Constraints)

### domain/
- MUST NOT import: `usecases/`, `adapters/`, `infrastructure/`, `app/`
- Contains:
  - Entities, Value Objects, Aggregates
  - Domain Services (pure business logic only)
  - Domain Events
- MUST:
  - Contain no framework, transport, persistence, or IO code
  - Remain persistence-ignorant
  - Express business invariants only

### usecases/
- MAY import: `domain/`
- MUST NOT import: `adapters/`, `infrastructure/`, `app/`
- Contains:
  - Use Case input ports (interfaces)
  - Application Services / Interactors
  - Input / Output DTOs
  - Application-owned ports
  - Transaction boundary coordination
- MUST:
  - Depend only on abstractions
  - Contain no transport or persistence implementations

### adapters/
- MAY import: `usecases/`
- MUST NOT import: `infrastructure/`, `app/`
- Contains:
  - Controllers / Resolvers / CLI handlers / MQ consumers
  - Presenters / Response mappers
- MUST:
  - Translate protocol input → use case input DTO
  - Invoke use case input ports only
  - Use DTOs, not domain objects, at boundaries

### infrastructure/
- MAY import:
  - `domain/`
  - `domain/ports`
  - `usecases/ports`
- MUST NOT import:
  - `adapters/`, `app/`
  - Concrete use case implementations
- Contains:
  - Repository implementations
  - External API clients
  - Message publishers
  - Technical frameworks and drivers

### app/
- MAY import all layers
- Responsible ONLY for:
  - Dependency wiring
  - Configuration
  - Process startup
  - Lifecycle management
- MUST NOT contain business or application logic

---

## 3. Ports Ownership Rules

### domain/ports
Used for domain-owned contracts:
- Aggregate repositories
- Domain event publishers
- Domain policies

### usecases/ports
Used for application-owned dependencies:
- Notifications
- External workflow coordination
- Integration hooks
- Unit of Work / Transaction management
- Read-model gateways

Rules:
- Domain and usecases depend only on abstractions
- Infrastructure contains implementations only
- If unclear, choose the innermost valid layer

---

## 4. Use Case Contracts

- Every use case MUST expose an input port (interface)
- Input ports are defined in `usecases/`
- Adapters depend ONLY on these interfaces

### DTO Rules
- Input and output DTOs are defined in `usecases/`
- Adapters MUST NOT expose domain entities or value objects

---

## 5. Adapter Responsibilities

### Primary Adapters (`adapters/`)
- Handle inbound protocols (HTTP, GraphQL, CLI, MQ)
- Validate, authorize, map DTOs, invoke use cases

### Output Mapping
- Presenters / response mappers
- Convert output DTOs → protocol responses
- MUST NOT invoke use cases

### Secondary Adapters (`infrastructure/`)
- Persistence, external clients, publishers
- Technical policies (retry, timeout, serialization)

---

## 6. Transaction & Unit of Work

- Transaction boundaries are application concerns
- Interfaces live in `usecases/ports` or `domain/ports`
- Implementations live in `infrastructure/`
- Use cases coordinate transactions explicitly

---

## 7. Domain Service vs Application Service

- Domain Services:
  - Pure business logic only
  - No IO, no orchestration, no external calls
- Application Services:
  - Orchestrate aggregates
  - Call outbound ports
  - Manage transactions

---

## 8. CQRS Compatibility

### Command Path
- State changes MUST go through aggregates
- Enforce invariants and emit domain events

### Query Path
- Read models may bypass aggregates
- Must be side-effect free
- Must not live in `domain/`

---

## 9. Explicit Non-Goals

- Do NOT collapse layers
- Do NOT put technical code in `domain/` or `usecases/`
- Do NOT leak transport schemas into domain
- Do NOT break dependency inversion

---

## 10. Scaling & Packaging

For larger systems:

```text
contexts/<context_name>/
  - domain/
  - usecases/
  - adapters/
  - infrastructure/
```

---

## 11. Code Generation Expectations

- Show exact folder placement
- Respect all dependency constraints
- Keep contracts explicit and small
- Prefer clarity over brevity
- When unsure, preserve domain purity
