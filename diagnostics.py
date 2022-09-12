import matplotlib.pyplot as plt
import numpy as np

'''
This class writes the LOG file that saves outputs from Trinity.

This file also contains older "diagnostic" classes which are no longer used
'''

class ProfileSaver:

    def __init__(self):

        # saves TRINITY data as a nested dictionary
        log = {}

        # init profiles
        log['time']  = []
        log['n']     = []
        log['pi']    = []
        log['pe']    = []
        log['aLn']     = []
        log['aLpi']    = []
        log['aLpe']    = []
        log['Gamma'] = []
        log['Qi']    = []
        log['Qe']    = []

        # additional profiles
        log['fusion_rate']  = []
        log['P_fusion_Wm3'] = [] 
        log['P_brems_Wm3']  = [] 
        log['nu_ei_Hz']     = [] 

        # power balance
        log['power balance'] = {}
        pb = log['power balance']
        pb['force_n'] = []
        pb['force_pi'] = []
        pb['force_pe'] = []
        pb['Ei'] = []
        pb['Ee'] = []
        pb['Gi']    = []
        pb['Ge']    = []
        pb['Hi']    = []
        pb['He']    = []
        pb['P_fusion'] = []
        pb['P_brems'] = []
        pb['aux_source_n'] = []
        pb['aux_source_pi'] = []
        pb['aux_source_pe'] = []

        self.log = log

    def save(self,engine):

        # version
        self.log['version'] = engine.version
        
        # profile
        n  = engine.density.profile
        pi = engine.pressure_i.profile
        pe = engine.pressure_e.profile
        G  = engine.Gamma.profile
        Qi = engine.Qi.profile
        Qe = engine.Qe.profile
        Gi = engine.Gi.full.profile
        Ge = engine.Ge.full.profile
        Hi = engine.Hi.full.profile
        He = engine.He.full.profile
        t  = engine.time

        aLn  = - engine.density   .grad_log.profile # a / Ln
        aLpi = - engine.pressure_i.grad_log.profile # a / Lpi
        aLpe = - engine.pressure_e.grad_log.profile # a / Lpe

        self.log['time'].append(t)
        self.log['n'].append(n)
        self.log['pi'].append(pi)
        self.log['pe'].append(pe)
        self.log['Gamma'].append(G)
        self.log['Qi'].append(Qi)
        self.log['Qe'].append(Qe)

        self.log['aLn'] .append(aLn)
        self.log['aLpi'].append(aLpi)
        self.log['aLpe'].append(aLpe)

        self.log['fusion_rate'] .append( engine.fusion_rate ) 
        self.log['P_fusion_Wm3'].append( engine.P_fusion_Wm3 ) 
        self.log['P_brems_Wm3'] .append( engine.P_brems_Wm3 ) 
        self.log['nu_ei_Hz'].append( engine.nu_ei )

        # store power balance
        pb = self.log['power balance']
        en = engine
        pb['force_n']  .append( en.force_n  )
        pb['force_pi'] .append( en.force_pi )
        pb['force_pe'] .append( en.force_pe )
        pb['Ei']       .append( en.Ei )
        pb['Ee']       .append( en.Ee )
        pb['Gi']       .append(Gi)
        pb['Ge']       .append(Ge)
        pb['Hi']       .append(Hi)
        pb['He']       .append(He)
        pb['P_fusion'] .append( en.P_fusion )
        pb['P_brems']  .append( en.P_brems )
        pb['aux_source_n'] .append( en.aux_source_n )
        pb['aux_source_pi'].append( en.aux_source_pi )
        pb['aux_source_pe'].append( en.aux_source_pe )



    def store_system(self,engine):
    # saves settings for reproducing runs

        # time step info lives here
        time_settings = {}
        time_settings['alpha']    = engine.alpha
        time_settings['dtau']     = engine.dtau
        time_settings['N_steps']  = engine.N_steps
        time_settings['N_prints'] = engine.N_prints
        time_settings['model']    = engine.model
        self.log['system'] = time_settings

        # profile info lives here
        profile_settings = {}
        profile_settings['N_radial']   = engine.N_radial
        profile_settings['rho_edge']   = engine.rho_edge
        profile_settings['rho_inner']  = engine.rho_inner
        profile_settings['rho_axis']   = engine.rho_axis
        profile_settings['grho']       = engine.grho
        profile_settings['drho']       = engine.drho
        profile_settings['area']       = engine.area.profile
        profile_settings['n_core']     = engine.n_core  
        profile_settings['n_edge']     = engine.n_edge 
        profile_settings['pi_core']    = engine.pi_core
        profile_settings['pi_edge']    = engine.pi_edge
        profile_settings['pe_core']    = engine.pe_core
        profile_settings['pe_edge']    = engine.pe_edge
        profile_settings['source_n' ]   = engine.source_n
        profile_settings['source_pi']   = engine.source_pi
        profile_settings['source_pe']   = engine.source_pe

        profile_settings['source_model']   = engine.source_model
        if (engine.source_model == 'Gaussian'):

            profile_settings['Sn_height']  = engine.Sn_height  
            profile_settings['Spi_height'] = engine.Spi_height 
            profile_settings['Spe_height'] = engine.Spe_height 
            profile_settings['Sn_width']   = engine.Sn_width      
            profile_settings['Spi_width']  = engine.Spi_width   
            profile_settings['Spe_width']  = engine.Spe_width    
            profile_settings['Sn_center']  = engine.Sn_center  
            profile_settings['Spi_center'] = engine.Spi_center 
            profile_settings['Spe_center'] = engine.Spe_center 

        elif (engine.source_model == 'external'):

            profile_settings['ext_source_file'] = engine.ext_source_file


        self.log['profiles'] = profile_settings

        #
        self.store_normalizations(engine)
        
    def store_normalizations(self,engine):

        norms = {}

        n = engine.norms
        norms['t_ref'] = n.t_ref
        norms['p_ref'] = n.p_ref
        norms['n_ref'] = n.n_ref
        norms['T_ref'] = n.T_ref
        norms['B_ref'] = n.B_ref
        norms['a_ref'] = n.a_ref

        norms['vT_ref']                = n.vT_ref
        norms['gyro_scale']            = n.gyro_scale
        norms['pressure_source_scale'] = n.pressure_source_scale

        self.log['norms'] = norms

    def export(self, fout='trinity_log.npy'):
        np.save(fout, self.log)

