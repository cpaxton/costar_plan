-- This is a customization script. It is intended to be used to customize a scene in
-- various ways, mainly when simulation is not running. When simulation is running,
-- do not use customization scripts, but rather child scripts if possible

-- Variable sim_call_type is handed over from the system

-- DO NOT WRITE CODE OUTSIDE OF THE if-then-end SECTIONS BELOW!! (unless the code is a function definition)

if (sim_call_type==sim_customizationscriptcall_initialization) then
    -- this is called just after this script was created (or reinitialized)
    -- Do some initialization here

    -- By default we disable customization script execution during simulation, in order
    -- to run simulations faster:
    simSetScriptAttribute(sim_handle_self,sim_customizationscriptattribute_activeduringsimulation,false)
    -- simLoadModel('/Applications/V-REP_PRO_EDU_V3_4_0_Mac/models/robots/non-mobile/KUKA\ LBR\ iiwa\ 14\ R820.ttm')
end

if (sim_call_type==sim_customizationscriptcall_nonsimulation) then
    -- This is called on a regular basis when simulation is not running.
    -- This is where you would typically write the main code of
    -- a customization script
end

if (sim_call_type==sim_customizationscriptcall_lastbeforesimulation) then
    -- This is called just before a simulation starts
end

if (sim_call_type==sim_customizationscriptcall_simulationactuation) then
    -- This is called by default from the main script, in the "actuation" phase.
    -- but only if you have previously not disabled this script to be active during
    -- simulation (see the script's initialization code above)
end

if (sim_call_type==sim_customizationscriptcall_simulationsensing) then
    -- This is called by default from the main script, in the "sensing" phase,
    -- but only if you have previously not disabled this script to be active during
    -- simulation (see the script's initialization code above)
end

if (sim_call_type==sim_customizationscriptcall_simulationpausefirst) then
    -- This is called just after entering simulation pause
end

if (sim_call_type==sim_customizationscriptcall_simulationpause) then
    -- This is called on a regular basis when simulation is paused
end

if (sim_call_type==sim_customizationscriptcall_simulationpauselast) then
    -- This is called just before leaving simulation pause
end

if (sim_call_type==sim_customizationscriptcall_firstaftersimulation) then
    -- This is called just after a simulation ended
end

if (sim_call_type==sim_customizationscriptcall_lastbeforeinstanceswitch) then
    -- This is called just before an instance switch (switch to another scene)
end

if (sim_call_type==sim_customizationscriptcall_firstafterinstanceswitch) then
    -- This is called just after an instance switch (switch to another scene)
end

if (sim_call_type==sim_customizationscriptcall_cleanup) then
    -- this is called just before this script gets destroyed (or reinitialized)
    -- Do some clean-up here
end
