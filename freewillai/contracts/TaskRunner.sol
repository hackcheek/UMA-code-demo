// SPDX-License-Identifier: OTHER
pragma solidity ^0.8.9;

import "contracts/Utils.sol";
import "contracts/FreeWillAIToken.sol";


struct Task {
    string modelUrl;
    string datasetUrl;
    string resultUrl;
    mapping(bytes32 => uint) resultCounts;
    Result[] results;
    uint256 startTime;
    uint reward;
    address sender;
    uint minTime;
    uint maxTime;
    uint minResults;
}

struct Result {
    string resultUrl;
    address node;
    uint stake;
}

contract TaskRunner {
    FreeWillAI public token;
    Utils utils = new Utils();
    Task[] public availableTasks;
    mapping(bytes32 => string) public resultsMap;
    mapping (address => uint) public stakingAmounts;
    uint public stakingMinimum = 100;
    uint public taskPrice = 10;
    uint consensusThreshold = 50; // 50%

    constructor(address _address){
        token = FreeWillAI(_address);
    }
    
    /*
    @param modelUrl : IPFS url where the AI model are
    @param datasetUrl : IPFS url where the dataset are
    @param minTime : Minimum time to wait for results
    @param maxTime : Maximum time to wait for results
    @param minResults : Minimum required results. This is proportional to the security level of the run
    */
    function addTask(string memory modelUrl, string memory datasetUrl, uint minTime, uint maxTime, uint minResults) public {
        require(token.balanceOf(msg.sender) >= taskPrice, "Not enough FWAI tokens to add task.");
        require(minTime < maxTime, "Bad arguments: minTime must be less than maxTime");
        require(1 < minResults, "Bad arguments: minResults must be more than 1");
        token.transferFrom(msg.sender, address(this), taskPrice);

        Task storage task = availableTasks.push();
        task.modelUrl = modelUrl;
        task.datasetUrl = datasetUrl;
        task.startTime = block.timestamp;
        task.reward = taskPrice;
        task.sender = msg.sender;
        task.minTime = minTime;
        task.maxTime = maxTime;
        task.minResults = minResults;

        emit TaskAdded(availableTasks.length - 1, modelUrl, datasetUrl);
    }
    event TaskAdded(
        uint indexed taskIndex,
        string modelUrl,
        string datasetUrl
    );

    function isValidated(uint taskIndex) public view returns (bool) {
        return !(utils.equalStrings(availableTasks[taskIndex].resultUrl, ""));
    }

    function getAvailableTasksCount() public view returns (uint) {
        return availableTasks.length;
    }

    function getAvailableTaskResults(uint taskIndex) public view returns (Result[] memory) {
        return availableTasks[taskIndex].results;
    }
        
    function getAvailableTaskResultsCount(uint taskIndex) public view returns (uint) {
        return availableTasks[taskIndex].results.length;
    }

    function getAvailableTask(uint taskIndex) public view returns (string memory, string memory) {
        require(taskIndex < availableTasks.length, "Invalid index");
        Task storage task = availableTasks[taskIndex];
        return (task.modelUrl, task.datasetUrl);
    }

    function submitResult(uint taskIndex, string calldata modelUrl, string calldata datasetUrl, string calldata resultUrl) public {
        require(taskIndex < availableTasks.length, "Task doesn't exist. taskIndex too high");
        require(!isInTimeout(taskIndex), "Submitting outside of this task's time window. Too late");
        require(checkStakingEnough(), "Your stake is not high enough to submit a result");
        Task storage task = availableTasks[taskIndex];
        require(utils.equalStrings(task.modelUrl, modelUrl), "modelUrl doesn't match the task on that index");
        require(utils.equalStrings(task.datasetUrl, datasetUrl), "datasetUrl doesn't match the task on that index");
        
        Result memory result = Result(resultUrl, msg.sender, stakingMinimum);
        stakingAmounts[msg.sender] -= stakingMinimum;
        
        task.results.push(result);

        uint resultCounts = getAvailableTaskResultsCount(taskIndex);
        emit TaskSubmitted(taskIndex, resultUrl, resultCounts, msg.sender);
    }
    event TaskSubmitted(
        uint indexed taskIndex,
        string resultUrl,
        uint resultCounts,
        address sender
    );

    function validateAllTasksIfReady() public{
        for(uint i = 0; i<availableTasks.length; i++){
            validateTaskIfReady(i);
        }
    }
    
    function stake(uint amount) public{
        token.transferFrom(msg.sender, address(this), amount);
        stakingAmounts[msg.sender] += amount;
    }

    function unstake(uint amount) public{
        require(stakingAmounts[msg.sender] >= amount, "Not enough tokens staked");
        token.approve(address(this), amount);
        token.transfer(msg.sender, amount);
        stakingAmounts[msg.sender] -= amount;
    }

    function getTaskResult(string calldata modelUrl, string calldata datasetUrl) public view returns (string memory){
        bytes32 taskHash = utils.hash2(modelUrl, datasetUrl);
        return resultsMap[taskHash];
    }

    function checkStakingEnough() view public returns (bool){
        // require(msg.sender.balance > stakingMinimum);
        return stakingAmounts[msg.sender] >= stakingMinimum;
    }

    function isInTimeout(uint taskIndex) view public returns (bool){
        Task storage task = availableTasks[taskIndex];
        return (block.timestamp > task.startTime + task.maxTime);
    }

    function checkIfReadyToValidate(uint taskIndex) view public returns (bool){
        Task storage task = availableTasks[taskIndex];
        return (
            task.startTime + task.minTime <= block.timestamp 
            && block.timestamp <= task.startTime + task.maxTime 
            && task.minResults <= getAvailableTaskResultsCount(taskIndex)
        );
    }

    function getTaskTimeLeft(uint taskIndex) view public returns (int){
        Task storage task = availableTasks[taskIndex];
        return int(task.startTime + task.maxTime) - int(block.timestamp);
    }
    function getTimestamp() view public returns (uint){
        return block.timestamp;
    }
    function getblocktime() private pure returns (uint256 result){
        return 0 ;
    }

    function validateTaskIfReady(uint taskIndex) public{
        Task storage task = availableTasks[taskIndex];
        getblocktime();
        if(checkIfReadyToValidate(taskIndex) && utils.isEmptyString(task.resultUrl)){
            task.resultUrl = getValidResult(taskIndex);
            bytes32 taskHash = utils.hash2(task.modelUrl, task.datasetUrl);
            resultsMap[taskHash] = task.resultUrl;
            rewardAndPunishNodes(taskIndex);
        } else if (isInTimeout(taskIndex)) {
            // Payment return to user
            token.approve(address(this), taskPrice);
            token.transfer(task.sender, taskPrice);
        }

    }
    

    function getValidResult(uint taskIndex) internal returns (string memory){
        Task storage task = availableTasks[taskIndex];
        Result[] memory results = task.results;
        string memory mostPopularResult;
        uint mostPopularResultCount = 0;
        uint consensusNeeded = (results.length * consensusThreshold) / 100;
        for(uint i = 0; i < results.length; i++){
            string memory resultUrl = results[i].resultUrl;
            bytes32 resultUrlHash = utils.hash(resultUrl);
            task.resultCounts[resultUrlHash] += 1;

            if(task.resultCounts[resultUrlHash] > mostPopularResultCount){
                mostPopularResult = resultUrl;
                mostPopularResultCount = task.resultCounts[resultUrlHash];
            }
        }
        if(mostPopularResultCount >= consensusNeeded){
            return mostPopularResult;
        }else{
            return "";
        }
    }
    
    function rewardAndPunishNodes(uint taskIndex) internal{
        Task storage task = availableTasks[taskIndex];
        Result[] memory results = task.results;
        uint totalStake = 0;
        uint totalCorrect = 0;
        for(uint i = 0; i < results.length; i++){
            totalStake += results[i].stake;
            if(utils.equalStrings(results[i].resultUrl, task.resultUrl)){
                totalCorrect++;
            }
        }
        uint consensusNeeded = (results.length * consensusThreshold) / 100;
        if(totalCorrect > consensusNeeded) {
            uint totalReward = totalStake + task.reward;
            for(uint i = 0; i < results.length; i++){
                if(utils.equalStrings(results[i].resultUrl, task.resultUrl)){
                    stakingAmounts[results[i].node] += totalReward / totalCorrect;
                }
            }
        }
        else {
            // Return stake to nodes
            for(uint i = 0; i < results.length; i++){
                stakingAmounts[results[i].node] += stakingMinimum;
            }
            // Payment return to user
            token.approve(address(this), taskPrice);
            token.transfer(task.sender, taskPrice);
       }
    }
}
    
