class Time():

    def __init__(self, inputs):

        # read time parameters from input file

        # the following are being deprecated from 'grid' and replaced in 'time', but are included here for backwards compat
        grid_parameters = inputs.get('grid', {})
        self.alpha = grid_parameters.get('alpha', 1)
        self.dtau = grid_parameters.get('dtau', 0.5)
        self.N_steps = grid_parameters.get('N_steps', 1000)

        time_parameters = inputs.get('time', {})
        self.max_newton_iter = time_parameters.get('max_newton_iter', 4)
        self.newton_threshold = time_parameters.get('newton_threshold', 2.0)
        # these "time" settings succeed the "grid" settings above, keeping both now for backwards compatibility
        self.alpha = time_parameters.get('alpha', self.alpha)
        self.dtau = time_parameters.get('dtau', self.dtau)
        self.N_steps = time_parameters.get('N_steps', self.N_steps)

        # init some variables
        self.time   = 0
        self.t_idx  = 0
        self.gx_idx = 0
        self.p_idx  = 0
        self.prev_p_id = 0
        self.newton_mode = False
