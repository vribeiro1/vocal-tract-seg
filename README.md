# Automatic Segmentation of Vocal Tract Articulators in Real-Time Magnetic Resonance Imaging

<hr>

This repository contains the code related to the following studies:

<ul>

<li>
<b>Deep Supervision of the Vocal Tract Shape for Articulatory Synthesis of Speech</b><br>
Vinicius Ribeiro
Ph.D. Thesis
</li>

<li>
<b>Automatic Segmentation of Vocal Tract Articulators in Real-Time Magnetic Resonance Imaging</b><br>
Vinicius Ribeiro, Karyna Isaieva, Justine Leclere, Jacques Felblinger, Pierre-André Vuissoz, Yves Laprie<br>
Nov 10, 2023 <a href="https://vribeiro1.github.io/publications#:~:text=Computer%20Methods%20and%20Programs%20in%20Biomedicine">Computer Methods and Programs in Biomedicine</a><br>
</li>

</ul>

<br>

<hr>

# External dependencies

This repository requires vt_tracker and vt_tools. They are available at Inria's Gitlab
(<a href="https://gitlab.inria.fr/vsouzari/vt_tracker">vt_tracker</a> and
<a href="https://gitlab.inria.fr/vsouzari/vt_tools">vt_tools</a>) . To install the library, follow
the instructions bellow.

<ol>

<li>Clone the repos</li>

```
>>> git clone git@gitlab.inria.fr:vsouzari/vt_tools.git
>>> git clone git@gitlab.inria.fr:vsouzari/vt_tracker.git
```

<li>Install the repo</li>

```
>>> pip3 install -e /path/to/vt_tools
>>> pip3 install -e /path/to/vt_tracker
```

</ol>
