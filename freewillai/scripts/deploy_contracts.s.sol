pragma solidity ^0.8.13;

import "forge-std/Script.sol";
import "../contracts/TaskRunner.sol";
import "../contracts/FreeWillAIToken.sol";

contract Deploy is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        vm.startBroadcast(deployerPrivateKey);

        FreeWillAI token = new FreeWillAI();
        TaskRunner taskRunner = new TaskRunner(address(token));

        vm.stopBroadcast();
    }
}
