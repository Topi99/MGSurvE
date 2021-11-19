# MGSurvE: Mosquito Gene SurveillancE


MGSurvE is a project oriented towards the optimization of traps' placement in complex heterogeneous landscapes in an effort to minimize the time to detection of genetic variants of interest.


**Under Construction** :construction: **Please check back in a couple of weeks!**



![Git Build](https://github.com/Chipdelmal/MGSurvE/actions/workflows/main.yml/badge.svg)[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)![Travis Build](https://app.travis-ci.com/Chipdelmal/MGSurvE.svg?branch=main)

![landscape](https://github.com/Chipdelmal/MGSurvE/raw/main/img/demo.jpg)


Please have a look at the [documentation](https://chipdelmal.github.io/MGSurvE/) for more info and our [pypi](https://pypi.org/project/MGSurvE/) package for installation and versions.


[<img src="https://pypi.org/static/images/logo-large.6bdbb439.svg" height="50px" align="middle">](https://pypi.org/project/MGSurvE/)
# Progress and Features

* Setup pkg skeleton
  * ~~Setup pypi pkg~~
  * ~~Setup CI~~
  * ~~Setup autodoc~~
* Landscape generation and variables update
  * ~~Basic landscape and matrices init~~
  * ~~Auto-calculate distance matrix~~
  * ~~Auto-calculate migration matrix~~
  * ~~Landscape onbjects constructors~~
  * ~~Auto distance and migration updates on change~~
  * ~~Update traps migrations in place (avoids memory shifts)~~
  * ~~Update traps migrations with new array if traps number changes~~
* Plots
  * Landscape
    * ~~Auto-assign markers with point-type~~
    * ~~Auto-assign colors with trap-type~~
    * ~~Auto-assign edge to trap if non-movable~~
    * ~~Auto-plot trap radii~~
    * Spherical coordinates projection
  * ~~Migration matrices~~
  * ~~Block Migration matrices~~
  * ~~Remove frames and axes~~
* Genetic Algorithm
  * Init chromosome
  * Fitness function
    * ~~Canonical form~~
    * ~~Get steady absorbing states~~
    * Put the cost function together
    * ~~Markov's fundamental matrix with no re-ordering~~
  * Skip non-movable traps
* Code efficiency
  * ~~Process traps movement in place~~
* Tests
  * ~~Points and traps numbers are updated correctly~~
  * ~~Check matrices for Markov validity~~
  * Test landscape with external migrations
  * Changing number of traps doesn't affect migration part
* Whishlist
  * Non-movable traps
    * ~~Add to object properties~~
  * ~~Different trap types~~
  * Male/Female kernel
  * Point-processes generation
  * Parallelize fitness function evaluation


# Authors and Funders

<img src="https://raw.githubusercontent.com/Chipdelmal/pyMSync/master/media/pusheen.jpg" height="130px" align="middle"><br>

* Lead and Dev: [Héctor M. Sánchez C.](https://chipdelmal.github.io/blog/)
* Devs: Elijah Bartolome
* PIs: [David L. Smith](http://www.healthdata.org/about/david-smith), [John M. Marshall](https://publichealth.berkeley.edu/people/john-marshall/)

<img src="https://github.com/Chipdelmal/MGSurvE/raw/main/img/berkeley.jpg" height="25px"> &nbsp; <img src="https://github.com/Chipdelmal/MGSurvE/raw/main/img/IHME.jpg" height="25px"> &nbsp;<img src="https://github.com/Chipdelmal/MGSurvE/raw/main/img/UCIMI.png" height="25px"> &nbsp;  <img src="https://github.com/Chipdelmal/MGSurvE/raw/main/img/IGI.png" height="25px"> &nbsp; <img src="https://github.com/Chipdelmal/MGSurvE/raw/main/img/DARPA.jpg" height="25px"> &nbsp; <img src="https://github.com/Chipdelmal/MGSurvE/raw/main/img/gates.jpg" height="25px">

