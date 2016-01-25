#ifndef NETWORK_H
#define NETWORK_H

#include <iostream>
#include <memory>
#include <functional>
#include <thread>

#if defined(Bool)
#undef Bool
#endif
#if defined(True)
#undef True
#endif
#if defined(False)
#undef False
#endif

#include "Network/Address.h"
#include "Network/Device.h"
#include "Network/Protocol/Local.h"
#include "Network/Protocol/Minuit.h"

using namespace OSSIA;

class Network
{
public:
    Network();
    ~Network();

    // expose the application and a scene node to i-score
    void publication();

    // get the scene node
    std::shared_ptr<Node> getSceneNode();

    // set the simRunning boolean
    void setSimRunning(bool b);

private:
    std::shared_ptr<Protocol> _localProtocol;
    std::shared_ptr<Device> _localDevice;
    std::shared_ptr<Node> _localSceneNode;
    std::thread _networkThread;
    bool _simRunning;
};

#endif // NETWORK_H
