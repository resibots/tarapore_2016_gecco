# How do different encodings influence the performance of the MAP-Elites algorithm?

Source code for the GECCO paper 'How do different encodings influence the performance of the MAP-Elites algorithm?'

D. Tarapore, J. Clune, A. Cully, J.-B. Mouret. To appear in the proceedings of GECCO 2016 (open-access).

These experiments depends on:
- [Sferes2](http://github.com/sferes2/sferes2)
- [nn2, a module for sferes2](http://github.com/sferes2/n22)
- [Robdyn](http://github.com/resibots/robdyn)

Please refer to the documentation of Sferes2 (wiki) to compile sferes2 and be able to replicate the present experiments. Please note that:
- the `modules` should be in the 'modules' directory of sferes2
- the experiments (`exp`) should be in the exp directory of sferes2

The map-elites module is not the standard one (because this code is older than the release of the 'official' map-elites module). You will need to install robdyn and link the robdyn modules to the sferes2 module.


