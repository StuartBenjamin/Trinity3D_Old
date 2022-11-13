
class Physics():

    def __init__(self, inputs):

        physics_parameters = inputs.get('physics', inputs.get('debug', {}))
        self.collisions = physics_parameters.get('collisions', True)
        self.alpha_heating = physics_parameters.get('alpha_heating', True)
        self.bremstrahlung = physics_parameters.get('bremstrahlung', True)
        self.update_equilibrium = physics_parameters.get('update_equilibrium', False)
        self.turbulent_exchange = physics_parameters.get('turbulent_exchange', False)
        self.compute_surface_areas = physics_parameters.get('compute_surface_areas', True)
