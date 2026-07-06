# Roadmap

## v1.0 Milestone — Core Functionality & Security (in progress)

**Phases:** 1-2

### Phase 1: Foundation & Core Functionality
**Goal:** Build core CLI structure with basic commands and security hardening.

**Status:** Completed

### Phase 2: Advanced Features & Integration  
**Goal:** Implement advanced features for the mmcli tool while maintaining security measures established in Phase 1.

**Depends on:** Phase 1

**Plans:** 6 plans (3 TDD test creation, 3 documentation/config updates)

**Tasks:**
- [x] `info` command - Show supported devices, models, and presets
- [x] `analyze` command - Analyze project dataset contents
- [x] `recommend` command - Recommend models and FE presets  
- [x] `deploy` command - Handle deployment operations
- [ ] Security testing for all new features (REQ-TESTS-07)
- [ ] Documentation updates with security best practices

### Phase 3: Testing and Documentation
**Goal:** Comprehensive testing and documentation for the mmcli tool.

**Depends on:** Phase 2

**Tasks:**
- [ ] Unit tests for all components ( REQ-TESTS-07, REQ-TESTS-08, REQ-TESTS-10)
- [ ] Integration tests for full pipelines
- [ ] End-to-end testing with example datasets
- [ ] Comprehensive documentation

Plans:
- [x] 02-01-PLAN.md — Unit tests for info command (TDD)
- [x] 02-02-PLAN.md — Unit tests for analyze command (TDD)
- [x] 02-03-PLAN.md — Unit tests for recommend command (TDD)
- [x] 02-04-PLAN.md — Unit tests for deploy command (TDD)
- [x] 02-05-PLAN.md — Document environment variables in CLI help
- [x] 02-06-PLAN.md — Config file examples documentation
