import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Deep Learning',
      link: {
        type: 'doc',
        id: 'deep-learning/index',
      },
      items: [
        {
          type: 'category',
          label: 'Theory & Fundamentals',
          link: {
            type: 'doc',
            id: 'deep-learning/theory/neural-networks',
          },
          items: [
            'deep-learning/theory/neural-networks',
            'deep-learning/theory/activation-functions',
            'deep-learning/theory/optimization',
            'deep-learning/theory/regularization',
            'deep-learning/theory/loss-functions',
          ],
        },
        {
          type: 'category',
          label: 'Mathematics',
          link: {
            type: 'doc',
            id: 'deep-learning/mathematics/linear-algebra',
          },
          items: [
            'deep-learning/mathematics/linear-algebra',
            'deep-learning/mathematics/calculus',
            'deep-learning/mathematics/probability',
            'deep-learning/mathematics/information-theory',
          ],
        },
        {
          type: 'category',
          label: 'Infrastructure & Engineering',
          link: {
            type: 'doc',
            id: 'deep-learning/infrastructure/deployment',
          },
          items: [
            'deep-learning/infrastructure/deployment',
            'deep-learning/infrastructure/scalability',
            'deep-learning/infrastructure/mlops',
            'deep-learning/infrastructure/hardware',
          ],
        },
        {
          type: 'category',
          label: 'Interview Preparation',
          link: {
            type: 'doc',
            id: 'deep-learning/interviews/common-questions',
          },
          items: [
            'deep-learning/interviews/common-questions',
            'deep-learning/interviews/problem-solving',
            'deep-learning/interviews/system-design',
            'deep-learning/interviews/implementation',
          ],
        },
      ],
    },
  ],
};

export default sidebars;
