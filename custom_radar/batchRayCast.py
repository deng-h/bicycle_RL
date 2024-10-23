import pybullet as p
import time
import math
import pybullet_data

useGui = True

if useGui:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.loadURDF("samurai.urdf")

# 射线起点
rayFrom = []
# 射线终点
rayTo = []
rayIds = []

# 射线数量
numRays = 1024
# 射线长度
rayLen = 13
rayHitColor = [1, 0, 0]
rayMissColor = [0, 1, 0]

replaceLines = True

for i in range(numRays):
    rayFrom.append([0, 0, 1])
    rayTo.append([
        rayLen * math.sin(2. * math.pi * float(i) / numRays),
        rayLen * math.cos(2. * math.pi * float(i) / numRays),
        1
    ])

    if replaceLines:
        rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor))
    else:
        rayIds.append(-1)

if not useGui:
    timingLog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "rayCastBench.json")

numSteps = 10

if useGui:
    numSteps = 327680

for i in range(numSteps):
    p.stepSimulation()
    # for j in range(8):
    #     results = p.rayTestBatch(rayFrom, rayTo, j + 1)

    results = p.rayTestBatch(rayFrom, rayTo)

    if useGui:
        if not replaceLines:
            p.removeAllUserDebugItems()

        for i in range(numRays):
            # [0] 碰撞对象的UID
            hitObjectUid = results[i][0]

            if hitObjectUid < 0:
                hitPosition = [0, 0, 0]
                p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i])
            else:
                hitPosition = results[i][3]
                p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[i])

    # time.sleep(1. / 24.)

if not useGui:
    p.stopStateLogging(timingLog)