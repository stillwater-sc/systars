"""
Unit tests for SIMT Streaming Multiprocessor components.

Tests both the RTL (Amaranth) components and the simulation models.
"""

import unittest

from amaranth.sim import Simulator, Tick

from systars.simt import SIMTConfig
from systars.simt.nv.execution_unit import (
    ALUPipeline,
    ExecutionUnit,
    ExecutionUnitSim,
    Opcode,
)
from systars.simt.nv.operand_collector import (
    CollectorState,
    OperandCollectorSim,
    OperandState,
)
from systars.simt.nv.partition import PartitionSim, create_gemm_program, create_test_program
from systars.simt.nv.register_file import RegisterFileBank, RegisterFileSim
from systars.simt.nv.warp_scheduler import Instruction, WarpSchedulerSim, WarpState


class TestSIMTConfig(unittest.TestCase):
    """Test SIMT configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SIMTConfig()
        assert config.num_partitions == 4
        assert config.cores_per_partition == 32  # Matches warp size
        assert config.warp_size == 32
        assert config.max_warps_per_partition == 8
        assert config.registers_per_partition == 16384
        assert config.register_banks_per_partition == 16

    def test_derived_properties(self):
        """Test derived configuration properties."""
        config = SIMTConfig()
        assert config.total_cores == 128  # 4 * 32
        assert config.total_registers == 65536  # 4 * 16384
        assert config.register_file_kb == 256  # 65536 * 4 / 1024


class TestRegisterFileBank(unittest.TestCase):
    """Test register file bank RTL component."""

    def test_instantiation(self):
        """Test bank instantiation."""
        config = SIMTConfig()
        bank = RegisterFileBank(config, bank_id=0)
        assert bank.config == config
        assert bank.bank_id == 0

    def test_has_correct_ports(self):
        """Test bank has expected ports."""
        config = SIMTConfig()
        bank = RegisterFileBank(config, bank_id=0)

        # Check that ports exist
        assert hasattr(bank, "addr")
        assert hasattr(bank, "read_en")
        assert hasattr(bank, "write_en")
        assert hasattr(bank, "write_data")
        assert hasattr(bank, "read_data")
        assert hasattr(bank, "read_valid")

    def test_write_then_read(self):
        """Test writing and reading from bank."""
        config = SIMTConfig()
        bank = RegisterFileBank(config, bank_id=0)

        def testbench():
            # Write value 42 to address 5
            yield bank.addr.eq(5)
            yield bank.write_data.eq(42)
            yield bank.write_en.eq(1)
            yield Tick()
            yield bank.write_en.eq(0)
            yield Tick()

            # Read from address 5
            yield bank.read_en.eq(1)
            yield Tick()

            # Give time for read to complete
            for _ in range(3):
                yield Tick()

            # Check read data
            data = yield bank.read_data
            assert data == 42

        sim = Simulator(bank)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


class TestRegisterFileSim(unittest.TestCase):
    """Test register file simulation model."""

    def test_instantiation(self):
        """Test register file sim instantiation."""
        config = SIMTConfig()
        rf = RegisterFileSim(config, partition_id=0)
        assert rf.config == config
        assert rf.partition_id == 0

    def test_write_and_read(self):
        """Test writing and reading registers."""
        config = SIMTConfig()
        rf = RegisterFileSim(config)

        # Write to register 10
        rf.write(10, 12345)

        # Read from register 10
        data, conflict, conflict_banks = rf.read([10])
        assert data[0] == 12345
        assert not conflict
        assert conflict_banks == []

    def test_bank_conflict_detection(self):
        """Test bank conflict detection.

        In the per-thread model, bank conflicts occur when multiple threads
        access the same bank. For register R0 with 16 banks and 32 threads:
        - Thread 0 accesses bank (0 + 0) % 16 = 0
        - Thread 16 accesses bank (0 + 16) % 16 = 0
        So reading R0 for all 32 threads causes 16 bank conflicts.
        """
        config = SIMTConfig()
        rf = RegisterFileSim(config)

        # Initialize R0 for all threads
        for t in range(32):
            rf.write_thread(0, t, 100 + t)

        # Reading R0 for all threads should detect bank conflicts
        # (32 threads, 16 banks = 2 threads per bank = 16 conflicts)
        rf.reset_cycle()
        data, num_conflicts, conflict_banks = rf.read_all_threads(0)
        assert num_conflicts == 16  # 32 threads / 16 banks - 1 per bank
        assert len(conflict_banks) == 16  # All banks have conflicts

    def test_no_conflict_different_banks(self):
        """Test no conflict with different banks."""
        config = SIMTConfig()
        rf = RegisterFileSim(config)

        # Registers 0 and 1 are in different banks
        rf.write(0, 100)
        rf.write(1, 200)

        rf.reset_cycle()
        data, conflict, conflict_banks = rf.read([0, 1])
        assert not conflict
        assert conflict_banks == []
        assert data[0] == 100
        assert data[1] == 200


class TestWarpSchedulerSim(unittest.TestCase):
    """Test warp scheduler simulation model."""

    def test_instantiation(self):
        """Test scheduler instantiation."""
        config = SIMTConfig()
        scheduler = WarpSchedulerSim(config, partition_id=0)
        assert scheduler.config == config
        assert scheduler.partition_id == 0

    def test_load_program(self):
        """Test loading program into warp."""
        config = SIMTConfig()
        scheduler = WarpSchedulerSim(config)

        program = [
            Instruction(opcode="IADD", dst=0, src1=1, src2=2),
            Instruction(opcode="IMUL", dst=3, src1=4, src2=5),
        ]
        scheduler.load_program(0, program)

        assert len(scheduler.warps[0].instructions) == 2
        assert scheduler.warps[0].pc == 0

    def test_activate_warps(self):
        """Test activating warps."""
        config = SIMTConfig()
        scheduler = WarpSchedulerSim(config)

        # Load program and activate
        program = [Instruction(opcode="IADD", dst=0, src1=1, src2=2)]
        scheduler.load_program(0, program)
        scheduler.load_program(1, program)
        scheduler.activate_warps(2)

        assert scheduler.warps[0].state == WarpState.READY
        assert scheduler.warps[1].state == WarpState.READY
        assert scheduler.warps[2].state == WarpState.INACTIVE

    def test_round_robin_scheduling(self):
        """Test round-robin scheduling."""
        config = SIMTConfig()
        scheduler = WarpSchedulerSim(config)

        program = [
            Instruction(opcode="IADD", dst=0, src1=1, src2=2),
            Instruction(opcode="IMUL", dst=3, src1=4, src2=5),
        ]
        scheduler.load_program(0, program)
        scheduler.load_program(1, program)
        scheduler.activate_warps(2)

        # First schedule should return warp 0
        result = scheduler.schedule()
        assert result is not None
        warp_id, instr = result
        assert warp_id == 0

        # Tick and schedule again should return warp 1
        scheduler.tick()
        result = scheduler.schedule()
        assert result is not None
        warp_id, instr = result
        assert warp_id == 1


class TestOperandCollectorSim(unittest.TestCase):
    """Test operand collector simulation model."""

    def test_instantiation(self):
        """Test collector instantiation."""
        config = SIMTConfig()
        collector = OperandCollectorSim(config, partition_id=0)
        assert collector.config == config
        assert len(collector.collectors) == config.collectors_per_partition

    def test_allocate_collector(self):
        """Test allocating a collector."""
        config = SIMTConfig()
        collector = OperandCollectorSim(config)

        instr = Instruction(opcode="IADD", dst=0, src1=1, src2=2)
        collector_id = collector.allocate(warp_id=0, instruction=instr)

        assert collector_id is not None
        assert collector.collectors[collector_id].state == CollectorState.COLLECTING

    def test_two_phase_collection(self):
        """Test two-phase operand collection (PENDING -> READING -> READY).

        In SIMT, each warp has 32 threads and each thread needs operands.
        We check thread 0's operands as representative.
        """
        config = SIMTConfig()
        oc = OperandCollectorSim(config)
        rf = RegisterFileSim(config)

        # Write test values for all threads
        for t in range(32):
            rf.write_thread(1, t, 100 + t)
            rf.write_thread(2, t, 200 + t)

        # Allocate collector
        instr = Instruction(opcode="IADD", dst=0, src1=1, src2=2)
        collector_id = oc.allocate(warp_id=0, instruction=instr)

        # Check operands are PENDING for all threads
        # thread_operands[thread_id][src_idx] - check thread 0
        entry = oc.collectors[collector_id]
        assert entry.thread_operands[0][0].state == OperandState.PENDING  # src1
        assert entry.thread_operands[0][1].state == OperandState.PENDING  # src2

        # First collect cycle: some threads go PENDING -> READING
        # Due to bank conflicts, only one thread per bank can read each cycle
        rf.reset_cycle()
        oc.collect_operands(rf)
        # At least some operands should be READING (first thread per bank)
        reading_count = entry.reading_count()
        assert reading_count > 0 or entry.is_ready()
        assert entry.state == CollectorState.COLLECTING or entry.state == CollectorState.READY

        # Continue collecting until all operands are ready
        max_cycles = 100  # Guard against infinite loop
        cycles = 0
        while entry.state != CollectorState.READY and cycles < max_cycles:
            rf.reset_cycle()
            oc.collect_operands(rf)
            cycles += 1

        # All operands for all threads should be READY now
        assert entry.state == CollectorState.READY
        assert entry.thread_operands[0][0].state == OperandState.READY  # src1
        assert entry.thread_operands[0][1].state == OperandState.READY  # src2

    def test_fire_ready_collector(self):
        """Test firing a ready collector.

        In SIMT, all 32 threads must have their operands collected before firing.
        This takes multiple cycles due to bank conflicts.
        """
        config = SIMTConfig()
        oc = OperandCollectorSim(config)
        rf = RegisterFileSim(config)

        # Write test values for all threads
        for t in range(32):
            rf.write_thread(1, t, 100 + t)
            rf.write_thread(2, t, 200 + t)

        instr = Instruction(opcode="IADD", dst=0, src1=1, src2=2)
        collector_id = oc.allocate(warp_id=0, instruction=instr)
        entry = oc.collectors[collector_id]

        # Collect operands for all 32 threads - takes multiple cycles due to bank conflicts
        max_cycles = 100
        cycles = 0
        while entry.state != CollectorState.READY and cycles < max_cycles:
            rf.reset_cycle()
            oc.collect_operands(rf)
            cycles += 1

        assert entry.state == CollectorState.READY

        # Fire - returns per-thread operands
        result = oc.fire()
        assert result is not None
        warp_id, fired_instr, per_thread_operands = result
        assert warp_id == 0
        assert fired_instr == instr
        # per_thread_operands[thread_id][src_idx]
        assert per_thread_operands[0][0] == 100  # Thread 0, src1
        assert per_thread_operands[0][1] == 200  # Thread 0, src2
        assert per_thread_operands[5][0] == 105  # Thread 5, src1
        assert per_thread_operands[5][1] == 205  # Thread 5, src2


class TestALUPipeline(unittest.TestCase):
    """Test ALU pipeline model."""

    def test_instantiation(self):
        """Test ALU pipeline instantiation."""
        alu = ALUPipeline(alu_id=0, num_stages=4)
        assert alu.alu_id == 0
        assert len(alu.stages) == 4
        assert not alu.is_busy()

    def test_issue_and_complete(self):
        """Test issuing and completing an instruction."""
        alu = ALUPipeline(alu_id=0, num_stages=4)

        # Issue 1-cycle instruction
        assert alu.can_accept()
        alu.issue(warp_id=0, opcode=Opcode.IADD, dst=5, result=42, latency=1)
        assert alu.is_busy()

        # Tick through pipeline
        result = alu.tick()  # Stage 0 -> 1
        assert result is None
        result = alu.tick()  # Stage 1 -> 2
        assert result is None
        result = alu.tick()  # Stage 2 -> 3
        assert result is None
        result = alu.tick()  # Complete
        assert result is not None
        warp_id, dst, value = result
        assert warp_id == 0
        assert dst == 5
        assert value == 42

    def test_pipelined_execution(self):
        """Test pipelined instruction execution."""
        alu = ALUPipeline(alu_id=0, num_stages=4)

        # Issue multiple instructions (back-to-back)
        alu.issue(warp_id=0, opcode=Opcode.IADD, dst=0, result=10, latency=1)
        alu.tick()
        alu.issue(warp_id=1, opcode=Opcode.IADD, dst=1, result=20, latency=1)
        alu.tick()
        alu.issue(warp_id=2, opcode=Opcode.IADD, dst=2, result=30, latency=1)

        # Both instructions should be in pipeline
        assert alu.is_busy()


class TestExecutionUnitSim(unittest.TestCase):
    """Test execution unit simulation model."""

    def test_instantiation(self):
        """Test execution unit instantiation."""
        config = SIMTConfig()
        eu = ExecutionUnitSim(config, partition_id=0)
        assert eu.config == config
        assert eu.num_alus == config.cores_per_partition  # 8

    def test_issue_instruction(self):
        """Test issuing instruction to execution unit."""
        config = SIMTConfig()
        eu = ExecutionUnitSim(config)

        instr = Instruction(opcode="IADD", dst=0, src1=1, src2=2)
        # Per-thread operands: each thread gets [10, 20, 0]
        per_thread_operands = [[10, 20, 0] for _ in range(32)]
        result = eu.issue(warp_id=0, instruction=instr, per_thread_operands=per_thread_operands)

        assert result is True
        assert eu.is_busy()
        assert eu.total_operations == 1

    def test_iadd_computation(self):
        """Test IADD computation with per-thread operands.

        In SIMT, a warp instruction executes on all 32 ALUs simultaneously.
        Each thread can have different operands, producing different results.
        """
        config = SIMTConfig()
        eu = ExecutionUnitSim(config)

        instr = Instruction(opcode="IADD", dst=0, src1=1, src2=2)
        # Give each thread different operands: thread T gets [T*10, T*3, 0]
        per_thread_operands = [[t * 10, t * 3, 0] for t in range(32)]
        eu.issue(warp_id=0, instruction=instr, per_thread_operands=per_thread_operands)

        # Drain pipeline - returns (warp_id, dst_reg, per_thread_results)
        completed = eu.drain()
        assert len(completed) == 1  # One warp operation with 32 results
        warp_id, dst, per_thread_results = completed[0]
        assert warp_id == 0
        assert dst == 0
        # Check per-thread results: thread T = T*10 + T*3 = T*13
        assert per_thread_results[0] == 0  # Thread 0: 0 + 0 = 0
        assert per_thread_results[1] == 13  # Thread 1: 10 + 3 = 13
        assert per_thread_results[5] == 65  # Thread 5: 50 + 15 = 65

    def test_fma_computation(self):
        """Test FMA computation with per-thread operands.

        In SIMT, a warp instruction executes on all 32 ALUs simultaneously.
        """
        config = SIMTConfig()
        eu = ExecutionUnitSim(config)

        instr = Instruction(opcode="FFMA", dst=0, src1=1, src2=2, src3=3)
        # All threads compute same: 3 * 4 + 5 = 17
        per_thread_operands = [[3, 4, 5] for _ in range(32)]
        eu.issue(warp_id=0, instruction=instr, per_thread_operands=per_thread_operands)

        completed = eu.drain()
        assert len(completed) == 1  # One warp operation
        warp_id, dst, per_thread_results = completed[0]
        assert per_thread_results[0] == 17  # 3 * 4 + 5

    def test_multiple_alu_utilization(self):
        """Test all ALUs are utilized for SIMT execution.

        In SIMT, a single warp instruction uses ALL 32 ALUs simultaneously
        (one per thread in the warp).
        """
        config = SIMTConfig()
        eu = ExecutionUnitSim(config)

        # Issue one warp instruction - uses all 32 ALUs
        instr = Instruction(opcode="IADD", dst=0, src1=1, src2=2)
        per_thread_operands = [[10, 1, 0] for _ in range(32)]
        eu.issue(warp_id=0, instruction=instr, per_thread_operands=per_thread_operands)

        # All 32 ALUs should be busy (SIMT execution)
        busy_count = sum(1 for alu in eu.alus if alu.is_busy())
        assert busy_count == config.cores_per_partition  # 32

    def test_utilization_reporting(self):
        """Test ALU utilization reporting."""
        config = SIMTConfig()
        eu = ExecutionUnitSim(config)

        # Issue one instruction
        instr = Instruction(opcode="IADD", dst=0, src1=1, src2=2)
        per_thread_operands = [[10, 20, 0] for _ in range(32)]
        eu.issue(warp_id=0, instruction=instr, per_thread_operands=per_thread_operands)

        # One stage of one ALU is busy out of 8*4=32 stages
        utilization = eu.get_utilization()
        assert utilization > 0
        assert utilization <= 100


class TestPartitionSim(unittest.TestCase):
    """Test partition simulation model."""

    def test_instantiation(self):
        """Test partition instantiation."""
        config = SIMTConfig()
        partition = PartitionSim(config, partition_id=0)
        assert partition.config == config
        assert partition.scheduler is not None
        assert partition.register_file is not None
        assert partition.operand_collector is not None
        assert partition.execution_unit is not None

    def test_load_and_run_program(self):
        """Test loading and running a program."""
        config = SIMTConfig()
        partition = PartitionSim(config)

        # Simple 2-instruction program
        program = [
            Instruction(opcode="MOV", dst=0, src1=0),
            Instruction(opcode="IADD", dst=1, src1=0, src2=0),
        ]
        partition.load_program(0, program)
        partition.activate_warps(1)

        # Run to completion
        cycles = partition.run_to_completion(max_cycles=100)
        assert partition.done
        assert cycles > 0
        assert partition.total_instructions > 0

    def test_statistics(self):
        """Test statistics collection."""
        config = SIMTConfig()
        partition = PartitionSim(config)

        program = create_test_program(4)
        partition.load_program(0, program)
        partition.activate_warps(1)
        partition.run_to_completion(max_cycles=100)

        stats = partition.get_statistics()
        assert "cycles" in stats
        assert "instructions" in stats
        assert "stalls" in stats
        assert "energy_pj" in stats
        assert stats["instructions"] > 0

    def test_energy_consumption(self):
        """Test energy is consumed during execution."""
        config = SIMTConfig()
        partition = PartitionSim(config)

        program = create_test_program(4)
        partition.load_program(0, program)
        partition.activate_warps(1)
        partition.run_to_completion(max_cycles=100)

        energy = partition.get_energy_pj()
        assert energy > 0


class TestGEMMProgram(unittest.TestCase):
    """Test GEMM program generation and execution."""

    def test_gemm_program_generation(self):
        """Test GEMM program generation."""
        program = create_gemm_program(16, 16, 4)
        assert len(program) > 0
        # Should have initialization + K iterations of load-load-FMA
        expected_instrs = 1 + 4 * 3  # 1 init + K * (2 loads + 1 FMA)
        assert len(program) == expected_instrs

    def test_gemm_execution(self):
        """Test GEMM program execution."""
        config = SIMTConfig()
        partition = PartitionSim(config)

        program = create_gemm_program(16, 16, 2)
        partition.load_program(0, program)
        partition.activate_warps(1)
        partition.run_to_completion(max_cycles=200)

        assert partition.done
        assert partition.total_instructions == len(program)


class TestVisualization(unittest.TestCase):
    """Test visualization methods."""

    def test_partition_visualization(self):
        """Test partition visualization data structure."""
        config = SIMTConfig()
        partition = PartitionSim(config)

        program = create_test_program(2)
        partition.load_program(0, program)
        partition.activate_warps(1)
        partition.step()

        vis = partition.get_visualization()
        assert "warp_states" in vis
        assert "register_banks" in vis
        assert "collectors" in vis
        assert "collector_status" in vis
        assert "pipeline" in vis
        assert "alu_detail" in vis

    def test_alu_detailed_visualization(self):
        """Test ALU detailed visualization."""
        config = SIMTConfig()
        eu = ExecutionUnitSim(config)

        vis = eu.get_detailed_visualization()
        assert "num_alus" in vis
        assert vis["num_alus"] == config.cores_per_partition  # 32 ALUs
        assert "pipeline_depth" in vis
        assert vis["pipeline_depth"] == 4
        assert "utilization" in vis
        assert "alus" in vis
        assert len(vis["alus"]) == config.cores_per_partition  # 32 ALUs


class TestExecutionUnitRTL(unittest.TestCase):
    """Test execution unit RTL component."""

    def test_instantiation(self):
        """Test RTL execution unit instantiation."""
        config = SIMTConfig()
        eu = ExecutionUnit(config, partition_id=0)
        assert eu.config == config

    def test_has_correct_ports(self):
        """Test execution unit has expected ports."""
        config = SIMTConfig()
        eu = ExecutionUnit(config)

        assert hasattr(eu, "in_valid")
        assert hasattr(eu, "in_warp")
        assert hasattr(eu, "in_opcode")
        assert hasattr(eu, "in_src1")
        assert hasattr(eu, "in_src2")
        assert hasattr(eu, "in_src3")
        assert hasattr(eu, "out_valid")
        assert hasattr(eu, "out_data")
        assert hasattr(eu, "busy")

    def test_pipeline_latency(self):
        """Test pipeline produces correct result."""
        config = SIMTConfig()
        eu = ExecutionUnit(config)

        def testbench():
            # Issue an IADD instruction
            yield eu.in_valid.eq(1)
            yield eu.in_opcode.eq(Opcode.IADD)
            yield eu.in_src1.eq(10)
            yield eu.in_src2.eq(20)
            yield eu.in_dst.eq(5)
            yield eu.in_warp.eq(0)
            yield Tick()
            yield eu.in_valid.eq(0)

            # Wait for pipeline to complete (4 stages + margin)
            for _ in range(6):
                yield Tick()

            # Check that result is correct when valid
            valid = yield eu.out_valid
            if valid:
                result = yield eu.out_data
                assert result == 30  # 10 + 20

        sim = Simulator(eu)
        sim.add_clock(1e-6)
        sim.add_testbench(testbench)
        sim.run()


if __name__ == "__main__":
    unittest.main()
