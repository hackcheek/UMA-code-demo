pragma solidity ^0.8.9;

import "forge-std/Test.sol";
import "contracts/TaskRunner.sol";
import "contracts/FreeWillAIToken.sol";

contract TestTaskRunner is Test {
    mapping (uint => address) nodesMap;

    TaskRunner taskRunner;
    FreeWillAI token;
    address tokenOwner;
    
    string testModelUrl;
    string testDatasetUrl;
    string testTrueResultUrl;

    uint minTime;
    uint maxTime;
    uint minResults;
    
    uint taskTimeWindow;
    uint stakingMinimum;
    uint taskReward;
    uint consensusThreshold;

    Task[] tasks;

    struct TaskType {
        string modelUrl;
        string datasetUrl;
        string resultUrl;
        mapping(bytes32 => uint) resultCounts;
        Result[] results;
        uint256 startTime;
        uint reward;
        address sender;
    }

    function setUp() public {
        token = new FreeWillAI();
        taskRunner = new TaskRunner(address(token));

        tokenOwner = vm.addr(99999);
        vm.prank(tokenOwner);
        token.initialize();

        testModelUrl = "http://ipfs.io/ipfs/QmZgvkWE6imotYLo3EwerUwiu6JwkuFHxZzJBQJH6aPz1t";
        testDatasetUrl = "http://ipfs.io/ipfs/QmbEkrZyNmwyiCFJisvwMFjxdURJpkn3Saw5QaZgoYWBdF";
        testTrueResultUrl = "http://ipfs.io/ipfs/QmcTVYwF5F9JFUdcikeuQCovALALYHh5abxGcfUuTH19Wx";
        
        // Defaults
        minTime = 1;
        maxTime = 200;
        minResults = 2;

        taskTimeWindow = 10;
        stakingMinimum = 100;
        taskReward = 10;
        consensusThreshold = 5; // 1 = 10%, 2 = 20%
    }

    function genNodes(uint numNodes) internal {
        for (uint i = 0; i<numNodes; i++){
            address publicKey = vm.addr(i+1);
            nodesMap[i+1] = publicKey;
        }
    }

    function test_noResultAtTask() public {
        genNodes(1);
        address user = nodesMap[1];

        vm.prank(tokenOwner);
        token.mint(user, taskReward);

        vm.warp(1);
        vm.startPrank(user);
        token.approve(address(taskRunner), taskReward);

        maxTime = 10;
        taskRunner.addTask(testModelUrl, testDatasetUrl, minTime, maxTime, minResults);
        vm.stopPrank();
        
        assertEq(token.balanceOf(user), 0);

        vm.warp(20);
        taskRunner.validateTaskIfReady(0);

        assertEq(token.balanceOf(user), taskReward);
    }
    
    function test_notValidTask() public {
        genNodes(3);
        address user = nodesMap[1];

        vm.prank(tokenOwner);
        token.mint(user, taskReward);

        vm.warp(1);
        vm.startPrank(user);
        token.approve(address(taskRunner), taskReward);
        taskRunner.addTask(testModelUrl, testDatasetUrl, minTime, maxTime, minResults);
        vm.stopPrank();
        
        for (uint i = 2; i < 4; i++) {
            address node = nodesMap[i];
            vm.prank(tokenOwner);
            token.mint(node, stakingMinimum);

            vm.startPrank(node);
            token.approve(address(taskRunner), stakingMinimum);

            assertEq(taskRunner.stakingAmounts(node), 0);
            taskRunner.stake(stakingMinimum);

            assertEq(token.balanceOf(node), 0);
            assertEq(taskRunner.stakingAmounts(node), stakingMinimum);

            taskRunner.submitResult(0, testModelUrl, testDatasetUrl, vm.toString(i));
            assertEq(taskRunner.stakingAmounts(node), 0);
            vm.stopPrank();
        }
        assertEq(token.balanceOf(user), 0);

        vm.warp(20);
        assertTrue(taskRunner.checkIfReadyToValidate(0));

        taskRunner.validateTaskIfReady(0);

        assertTrue(taskRunner.checkIfReadyToValidate(0));

        assertEq(token.balanceOf(user), taskReward);
        assertEq(taskRunner.stakingAmounts(nodesMap[2]), stakingMinimum);
        assertEq(taskRunner.stakingAmounts(nodesMap[3]), stakingMinimum);
    }

    function test_generateNodes() public {
        genNodes(6);

        emit log_address(nodesMap[1]);
        emit log_address(nodesMap[2]);
        emit log_address(nodesMap[3]);
        emit log_address(nodesMap[4]);
        emit log_address(nodesMap[5]);
        emit log_address(nodesMap[6]);
    }

    function test_AddTask() public {
        Task storage task = tasks.push();
        task.modelUrl = testModelUrl;
        task.datasetUrl = testDatasetUrl;
        task.startTime = block.timestamp;
        task.reward = taskReward;

        genNodes(1);
        address user = nodesMap[1];

        vm.prank(tokenOwner);
        token.mint(user, taskReward);

        vm.startPrank(user);
        token.approve(address(taskRunner), taskReward);
        taskRunner.addTask(testModelUrl, testDatasetUrl, minTime, maxTime, minResults);
        vm.stopPrank();
    }

    function test_AddTaskTimeout() public {
        // Generate user account
        genNodes(1);
        address user = nodesMap[1];

        // Minting user to run task
        vm.prank(tokenOwner);
        token.mint(user, taskReward);

        // Run task with maxTime=10
        vm.startPrank(user);
        token.approve(address(taskRunner), taskReward);
        taskRunner.addTask(testModelUrl, testDatasetUrl, minTime, 10 /*maxTime*/, minResults);
        vm.stopPrank();

        // Set time up 20
        vm.warp(20);

        // Bool asserts
        assertFalse(taskRunner.checkIfReadyToValidate(0));
        assertTrue(taskRunner.isInTimeout(0));
        // Validate task that should fail and return the spent to user
        taskRunner.validateTaskIfReady(0);
        assertEq(token.balanceOf(user), taskReward);
    }

    function test_AddTaskMinTimeAndMinResults() public {
        // Generate user account
        genNodes(5);
        address user = nodesMap[1];

        vm.warp(0);

        // Minting user to run task
        vm.prank(tokenOwner);
        token.mint(user, taskReward);

        // Run task with minTime=20
        vm.startPrank(user);
        token.approve(address(taskRunner), taskReward);
        taskRunner.addTask(testModelUrl, testDatasetUrl, 20 /*minTime*/, maxTime, minResults);
        vm.stopPrank();

        // Set time up 10
        vm.warp(10);

        // Bool asserts
        assertFalse(taskRunner.checkIfReadyToValidate(0));
        assertFalse(taskRunner.isInTimeout(0));

        // Validate task that should fail
        taskRunner.validateTaskIfReady(0);
        assertEq(token.balanceOf(user), 0);

        // Set time up 20
        vm.warp(20);

        // Bool asserts
        assertFalse(taskRunner.checkIfReadyToValidate(0));
        assertFalse(taskRunner.isInTimeout(0));

        // Validate task that should fail
        taskRunner.validateTaskIfReady(0);
        assertEq(token.balanceOf(user), 0);

        for (uint i = 2; i < 6; i++) {
            address node = nodesMap[i];
            vm.prank(tokenOwner);
            token.mint(node, stakingMinimum);

            vm.startPrank(node);
            token.approve(address(taskRunner), stakingMinimum);

            assertEq(taskRunner.stakingAmounts(node), 0);
            taskRunner.stake(stakingMinimum);

            assertEq(token.balanceOf(node), 0);
            assertEq(taskRunner.stakingAmounts(node), stakingMinimum);

            taskRunner.submitResult(0, testModelUrl, testDatasetUrl, testTrueResultUrl);
            assertEq(taskRunner.stakingAmounts(node), 0);

            if (i - 1 == 1) {
                assertFalse(taskRunner.checkIfReadyToValidate(0));
            } else {
                emit log_uint(i-1);
                assertTrue(taskRunner.checkIfReadyToValidate(0));
            }
            vm.stopPrank();
        }
        taskRunner.validateTaskIfReady(0);
        assertEq(taskRunner.getTaskResult(testModelUrl, testDatasetUrl), testTrueResultUrl);
    }

    function test_GetAvaialbleTasksCount() public view returns (uint) {
    }

    function test_GetAvaialbleTaskResultsCount(uint taskIndex) public view returns (uint) {
        
    }

    function test_GetAvailableTask(uint taskIndex) public view returns (string memory, string memory) {
        
    }

    function test_SubmitResult(uint taskIndex, string calldata modelUrl, string calldata datasetUrl, string calldata resultUrl) public {
        
    }

    function test_ValidateAllTasksIfReady() public{
        
    }
    
    function test_Stake() public{
        uint stakingAmount = 200;
        genNodes(1);
        address node1 = nodesMap[1];
        
        vm.prank(tokenOwner);
        token.mint(node1, stakingAmount);

        vm.startPrank(node1);
        token.approve(address(taskRunner), stakingAmount);
        taskRunner.stake(stakingAmount);
        vm.stopPrank();

        assertEq(taskRunner.stakingAmounts(node1), stakingAmount);
    }

    function test_Unstake(uint amount) public{
        
    }

    function test_GetTaskResult() public {
        vm.warp(1);
        uint stakingAmount = 200;
        genNodes(4);
        string memory modelUrl = "https://example.com/model";
        string memory datasetUrl = "https://example.com/dataset";
        string memory correctResultUrl = "https://example.com/correct-result";
        string memory incorrectResultUrl = "https://example.com/incorrect-result";
        
        address user = nodesMap[4];

        vm.prank(tokenOwner);
        token.mint(user, taskReward);
        
        vm.startPrank(user);
        token.approve(address(taskRunner), taskReward);
        taskRunner.addTask(modelUrl, datasetUrl, minTime, maxTime, minResults);
        vm.stopPrank();
        uint taskIndex = taskRunner.getAvailableTasksCount() - 1;

        for (uint i = 1; i < 4; i++) {
            vm.prank(tokenOwner);
            token.mint(nodesMap[i], stakingAmount);
            address currentNode = nodesMap[i];
            vm.startPrank(currentNode);
            token.approve(address(taskRunner), stakingAmount);
            if (i == 3){
                taskRunner.stake(stakingAmount);
                taskRunner.submitResult(taskIndex, modelUrl, datasetUrl, incorrectResultUrl);
                vm.stopPrank();
            }
            else{
                taskRunner.stake(stakingAmount);
                taskRunner.submitResult(taskIndex, modelUrl, datasetUrl, correctResultUrl);
                vm.stopPrank();
            }
        }

        vm.warp(20);
        assertTrue(taskRunner.checkIfReadyToValidate(taskIndex));
        taskRunner.validateTaskIfReady(taskIndex);
        assertEq(taskRunner.stakingAmounts(nodesMap[1]), stakingAmount+55);
        assertEq(taskRunner.stakingAmounts(nodesMap[2]), stakingAmount+55);
        assertEq(taskRunner.stakingAmounts(nodesMap[3]), stakingAmount-100);
    }

    function test_isValidated() public {
        vm.warp(1);
        uint stakingAmount = 200;
        genNodes(4);
        string memory modelUrl = "https://example.com/model";
        string memory datasetUrl = "https://example.com/dataset";
        string memory correctResultUrl = "https://example.com/correct-result";
        string memory incorrectResultUrl = "https://example.com/incorrect-result";
        
        address user = nodesMap[4];

        vm.prank(tokenOwner);
        token.mint(user, taskReward);
        
        vm.startPrank(user);
        token.approve(address(taskRunner), taskReward);
        taskRunner.addTask(modelUrl, datasetUrl, minTime, maxTime, minResults);
        
        vm.stopPrank();
        uint taskIndex = taskRunner.getAvailableTasksCount() - 1;

        console.logBool(taskRunner.isValidated(taskIndex));
        assertFalse(taskRunner.isValidated(taskIndex));

        for (uint i = 1; i < 4; i++) {
            vm.prank(tokenOwner);
            token.mint(nodesMap[i], stakingAmount);
            address currentNode = nodesMap[i];
            vm.startPrank(currentNode);
            token.approve(address(taskRunner), stakingAmount);
            if (i == 3){
                taskRunner.stake(stakingAmount);
                taskRunner.submitResult(taskIndex, modelUrl, datasetUrl, incorrectResultUrl);
                vm.stopPrank();
            }
            else{
                taskRunner.stake(stakingAmount);
                taskRunner.submitResult(taskIndex, modelUrl, datasetUrl, correctResultUrl);
                vm.stopPrank();
            }
        }
        vm.warp(20);
        taskRunner.validateTaskIfReady(taskIndex);
        assertTrue(taskRunner.isValidated(taskIndex));
    }

    function test_CheckStakingEnough() view public returns (bool){
        
    }


    function test_GetTaskTimeLeft() public {
        genNodes(1);
        address user = nodesMap[1];

        vm.prank(tokenOwner);
        token.mint(user, taskReward);

        vm.warp(0);
        vm.startPrank(user);
        token.approve(address(taskRunner), taskReward);

        maxTime = 10;
        taskRunner.addTask(testModelUrl, testDatasetUrl, minTime, maxTime, minResults);
        for (int i=0; i < int(maxTime + 1); i++) {
            vm.warp(uint(i));
            assertEq(taskRunner.getTaskTimeLeft(0), int(maxTime) - i);
        }
        vm.stopPrank();
    }

    function test_GetTimestamp() view public returns (uint){
        
    }

    function test_Getblocktime() private pure returns (uint256 result){
        
    }

    function test_ValidateTaskIfReady(uint taskIndex) public{
        
    }
    
    function test_GetValidResult(uint taskIndex) internal returns (string memory){
        
    }
    
    function test_RewardAndPunishNodes(uint taskIndex) internal{
        
    }
    function test_RewardAndPunishNodesTwoRightOneWrong(uint taskIndex) public{
        vm.warp(1);
        uint stakingAmount = 200;
        genNodes(4);
        string memory modelUrl = "https://example.com/model";
        string memory datasetUrl = "https://example.com/dataset";
        string memory correctResultUrl = "https://example.com/correct-result";
        string memory incorrectResultUrl = "https://example.com/incorrect-result";
        
        address user = nodesMap[4];

        vm.prank(tokenOwner);
        token.mint(user, taskReward);
        
        vm.startPrank(user);
        token.approve(address(taskRunner), taskReward);
        taskRunner.addTask(modelUrl, datasetUrl, minTime, maxTime, minResults);
        vm.stopPrank();
        taskIndex = taskRunner.getAvailableTasksCount() - 1;

        for (uint i = 1; i < 4; i++) {
            vm.prank(tokenOwner);
            token.mint(nodesMap[i], stakingAmount);
            address currentNode = nodesMap[i];
            vm.startPrank(currentNode);
            token.approve(address(taskRunner), stakingAmount);
            if (i == 3){
                taskRunner.stake(stakingAmount);
                taskRunner.submitResult(taskIndex, modelUrl, datasetUrl, incorrectResultUrl);
                vm.stopPrank();
            }
            else{
                taskRunner.stake(stakingAmount);
                taskRunner.submitResult(taskIndex, modelUrl, datasetUrl, correctResultUrl);
                vm.stopPrank();
            }
            }
        vm.warp(20);
        taskRunner.validateTaskIfReady(taskIndex);
        assertEq(taskRunner.stakingAmounts(nodesMap[1]), stakingAmount+55);
        assertEq(taskRunner.stakingAmounts(nodesMap[2]), stakingAmount+55);
        assertEq(taskRunner.stakingAmounts(nodesMap[3]), stakingAmount-100);
    }
}
