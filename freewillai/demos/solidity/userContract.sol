pragma solidity ^0.8.9;


Interface TaskRunner {
    function addTask(
        string memory modelUrl, 
        string memory datasetUrl, 
        uint minTime, 
        uint maxTime, 
        uint minResults
    ) external;
}


contract TestAsUser {
    FreeWillAI token;
    TaskRunner taskRunner;
    
    address tokenAddr;
    address taskRunnerAddr;

    string modelUrl;
    string promptUrl;

    uint minTime;
    uint maxTime;
    uint minResults;
    uint taskReward;

    constructor() public {
        fwaiContractAddr = 0xDAE95F004b4B308921c8fdead101555eAB83916B;
        fwaiContract = TaskRunner(fwaiContractAddr);

        // IPFS url where model is allocated
        modelUrl = 'https://ipfs.io/ipfs/QmYKJDJcpRSSBpmzbRsN587bBXAKZBYpy9MvySJXQD9GUv';
        promptUrl = 'https://ipfs.io/ipfs/QmabBaeQCcNHfa99jAN1tdgKRxkj1X7Rxzd7Trr5bj3CPY';
        
        // Defaults
        minTime = 1;
        maxTime = 200;
        minResults = 2;

        taskReward = 10;
    }
    
    function run() public {
        // token.approve(address(taskRunner), taskReward);
        taskRunner.addTask(modelUrl, promptUrl, minTime, maxTime, minResults);
    }
}
