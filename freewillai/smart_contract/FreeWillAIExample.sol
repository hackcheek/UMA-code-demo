pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";

contract MLTaskContract is Ownable {
    struct Task {
        bytes32 ipfsModelLink;
        bytes32 ipfsDatasetLink;
        bytes32 ipfsConfigLink;
        mapping(address => bytes32) nodeSubmissions;
        address[] nodes;
        mapping(bytes32 => uint) resultCounts;
    }

    Task[] public tasks;
    mapping(address => uint) public stakedAmounts;

    uint public nextTaskId = 1;
    uint public stakingAmount = 1 ether;
    uint public slashingAmount = 0.5 ether;
    uint public correctResultReward = 0.5 ether;
    uint public agreementThreshold = 50;

    event TaskSubmitted(uint indexed taskId, bytes32 result, address indexed node);
    event NodeStaked(address indexed node, uint amount);
    event NodeSlashed(address indexed node, uint amount);
    event NodeRewarded(address indexed node, uint amount);

    function submitTask(bytes32 ipfsModelLink, bytes32 ipfsDatasetLink, bytes32 ipfsConfigLink) public onlyOwner {
        tasks.push(Task(ipfsModelLink, ipfsDatasetLink, ipfsConfigLink));
    }

    function submitNodeResult(uint taskId, bytes32 result) public {
        Task storage task = tasks[taskId - 1];

        // Ensure node hasn't already submitted a result for this task
        require(task.nodeSubmissions[msg.sender] == bytes32(0), "Node already submitted result for this task");

        // Record the submission
        task.nodeSubmissions[msg.sender] = result;
        task.nodes.push(msg.sender);
        task.resultCounts[result] += 1;

        // Emit an event for the submission
        emit TaskSubmitted(taskId, result, msg.sender);
    }

    function rewardNode(uint taskId, bytes32 correctResult) internal {
        Task storage task = tasks[taskId - 1];

        // Calculate the reward per node
        uint rewardPerNode = correctResultReward / task.nodes.length;

        // Reward each node that submitted a correct result
        for (uint i = 0; i < task.nodes.length; i++) {
            address node = task.nodes[i];

            if (task.nodeSubmissions[node] == correctResult) {
                payable(node).transfer(rewardPerNode);

                // Emit an event for the reward
                emit NodeRewarded(node, rewardPerNode);
            }
        }
    }

    function slashNode(address node) internal {
        // Deduct the slashing amount from the node's staked amount
        uint amountToSlash = slashingAmount;
        if (stakedAmounts[node] < slashingAmount) {
            amountToSlash = stakedAmounts[node];
        }
        stakedAmounts[node] -= amountToSlash;

        // Transfer the slashed amount to the contract owner
        payable(owner()).transfer(amountToSlash);

        // Emit an event for the slashing
        emit NodeSlashed(node, amountToSlash);
    }

    function stake() public payable {
        // Require staked amount to be at least the required amount
        require(msg.value >= stakingAmount, "Insufficient staked amount");

        // Add the staked amount to the node's staked amount
        stakedAmounts[msg.sender] += msg.value;

        // Emit an event for the staking
        emit NodeStaked(msg.sender, msg.value);
    }

    function unstake() public {
        // Get the node's staked amount
        uint stakedAmount = stakedAmounts[msg.sender];

        // Require node to have a non-zero staked amount
        require(stakedAmount > 0, "No staked amount to withdraw");

        // Reset the node's staked amount
        stakedAmounts[msg.sender] = 0;

        // Transfer the staked amount back to the node
        payable(msg.sender).transfer(stakedAmount);
    }

    function reachConsensus(uint taskId) public onlyOwner {
        Task storage task = tasks[taskId - 1];

        bytes32 correctResult = findAgreedResult(taskId);

        if (correctResult != bytes32(0)) {
            rewardNode(taskId, correctResult);

            // Penalize nodes with incorrect submissions
            for (uint i = 0; i < task.nodes.length; i++) {
                address node = task.nodes[i];
                if (task.nodeSubmissions[node] != correctResult) {
                    slashNode(node);
                }
            }
        }
    }

    function findAgreedResult(uint taskId) internal view returns (bytes32) {
        Task storage task = tasks[taskId - 1];
        bytes32 mostCommonResult;
        uint highestCount = 0;

        for (uint i = 0; i < task.nodes.length; i++) {
            address node = task.nodes[i];
            bytes32 result = task.nodeSubmissions[node];
            uint count = task.resultCounts[result];

            if (count > highestCount) {
                highestCount = count;
                mostCommonResult = result;
            }
        }

        // Check if the agreement threshold is reached
        uint agreementPercentage = (highestCount * 100) / task.nodes.length;
        if (agreementPercentage >= agreementThreshold) {
            return mostCommonResult;
        } else {
            return bytes32(0);
        }
    }
}
