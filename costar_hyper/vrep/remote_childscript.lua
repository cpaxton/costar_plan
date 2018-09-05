--- Improved functions based on remoteApiCommandServerExample.ttt
displayText_function=function(inInts,inFloats,inStrings,inBuffer)
    -- Simply display a dialog box that prints the text stored in inStrings[1]:
    if #inStrings>=1 then
        simDisplayDialog('Message from the remote API client',inStrings[1],sim_dlgstyle_ok,false)
        return {},{},{'message was displayed'},'' -- return a string
    end
end

createDummy_function=function(inInts,inFloats,inStrings,inBuffer)
    -- Create a dummy object with specific name and coordinates
    if #inStrings>=1 and #inFloats>=3 then
        -- local dummyHandle=simGetObjectHandle(inStrings[1])
        -- if dummyHandle == -1 then
            dummyHandle=simCreateDummy(0.05)
        -- end
        local parent_handle=inInts[1]
        local errorReportMode=simGetInt32Parameter(sim_intparam_error_report_mode)
        simSetInt32Parameter(sim_intparam_error_report_mode,0) -- temporarily suppress error output (because we are not allowed to have two times the same object name)
        result = simSetObjectName(dummyHandle,inStrings[1])
        if result == -1 then
          simDisplayDialog('Setting object name failed',inStrings[1],sim_dlgstyle_ok,false)
        end
        simSetInt32Parameter(sim_intparam_error_report_mode,errorReportMode) -- restore the original error report mode
        simSetObjectPosition(dummyHandle,parent_handle,inFloats)
        if #inFloats>=7 then
            local orientation={unpack(inFloats, 4, 7)} -- get 4 quaternion entries from 4 to 7
            simSetObjectQuaternion(dummyHandle,parent_handle,orientation)
        end
        return {dummyHandle},{},{},'' -- return the handle of the created dummy
    end
end


createPointCloud_function=function(inInts,inFloats,inStrings,inBuffer)
    -- Create a dummy object with specific name and coordinates
    if #inStrings>=1 and #inFloats>=3 then

        -- The parent handle is the first integer parameter
        local parent_handle=inInts[1]

        -- Find an existing cloud with the specified name or create a new one
        simSetInt32Parameter(sim_intparam_error_report_mode,0) -- temporarily suppress error output (because we are not allowed to have two times the same object name)
        cloudHandle=simGetObjectHandle(inStrings[1])
        simSetInt32Parameter(sim_intparam_error_report_mode,errorReportMode) -- restore the original error report mode

        if cloudHandle ~= -1 then
            simRemoveObject(cloudHandle)
        end
        -- create a new cloud if none exists
        cloudHandle=simCreatePointCloud(0.01, 10, 0, 10)
        -- simDisplayDialog(('Call received! handle: ' .. cloudHandle),inStrings[1],sim_dlgstyle_ok,false)

        -- Update the name of the cloud
        local errorReportMode=simGetInt32Parameter(sim_intparam_error_report_mode)
        simSetInt32Parameter(sim_intparam_error_report_mode,0) -- temporarily suppress error output (because we are not allowed to have two times the same object name)
        result = simSetObjectName(cloudHandle,inStrings[1])
        if result == -1 then
          simDisplayDialog('Setting object name failed',inStrings[1],sim_dlgstyle_ok,false)
        end
        simSetInt32Parameter(sim_intparam_error_report_mode,errorReportMode) -- restore the original error report mode
        --- Set the position of the cloud relative to teh parent handle
        simSetObjectPosition(cloudHandle,parent_handle,inFloats)

        poseEntries=inInts[2]
        if #inFloats>=7 then
            local orientation={unpack(inFloats, 4, 7)} -- get 4 quaternion entries from 4 to 7
            simSetObjectQuaternion(cloudHandle,parent_handle,orientation)
        end
        -- print('pose vec quat:' .. {unpack(inFloats, 4, 7)})
        -- local cloud = simUnpackFloatTable(inStrings[2])
        cloudFloatCount=inInts[3]
        simAuxiliaryConsolePrint('cloudFloatCount: '..cloudFloatCount)
        pointBatchSize=30
        colorBatch=nil
        -- bit 1 is 1 so point clouds in cloud reference frame
        options = 1
        if #inStrings > 2 then
          -- bit 2 is 1 so each point is colored
          options = 3
          colors = simUnpackUInt8Table(inStrings[3])
        end
        -- Insert the point cloud points
        for i = 1, cloudFloatCount, pointBatchSize do
            startEntry=1+poseEntries+i
            local pointBatch={unpack(inFloats, startEntry, startEntry+pointBatchSize)}
            simAuxiliaryConsolePrint('threePoints:')

            simAuxiliaryConsolePrint(pointBatch[1])
            simAuxiliaryConsolePrint(pointBatch[2])
            simAuxiliaryConsolePrint(pointBatch[3])
            if #inStrings > 2 then
                colorBatch = {unpack(colors, startEntry, startEntry+pointBatchSize)}
            end

           simInsertPointsIntoPointCloud(cloudHandle, options, pointBatch, colors)
        end
        return {cloudHandle},{},{},'' -- return the handle of the created dummy
    end
end

executeCode_function=function(inInts,inFloats,inStrings,inBuffer)
    -- Execute the code stored in inStrings[1]:
    if #inStrings>=1 then
        return {},{},{loadstring(inStrings[1])()},'' -- return a string that contains the return value of the code execution
    end
end

if (sim_call_type==sim_childscriptcall_initialization) then
    simExtRemoteApiStart(19999)
end
